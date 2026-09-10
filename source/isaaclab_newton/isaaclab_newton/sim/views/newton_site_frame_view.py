# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-backed FrameView using Newton body labels and injected sites."""

from __future__ import annotations

import contextlib
import logging
import re
import sys

import warp as wp
from newton import ShapeFlags

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.physics import PhysicsEvent
from isaaclab.sim.views.base_frame_view import BaseFrameView
from isaaclab.sim.views.fabric_xform_selection import FabricXformSelection
from isaaclab.sim.views.xform_space_writer import FrameViewLocalSpaceWriter, FrameViewWorldSpaceWriter
from isaaclab.utils.string import resolve_matching_names
from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.warp import fabric as fabric_utils

from isaaclab_newton.physics.newton_manager import NewtonManager

logger = logging.getLogger(__name__)

WORLD_BODY_INDEX = -1

# Regex metacharacters that mark a body pattern as a genuine expression rather than a literal
# USD path. Patterns free of these can be resolved via an exact dict lookup instead of scanning
# every body label with a compiled regex.
_REGEX_TOKENS = frozenset(".*[]()+?|\\^$")


def _has_regex_tokens(pattern: str) -> bool:
    """Return whether ``pattern`` contains regex metacharacters (i.e. is not a literal path)."""
    return any(token in _REGEX_TOKENS for token in pattern)


# One resolved site registration: (body_patterns, local transform, xform scale, per_world, env_ids,
# destination prim paths).  The prim paths follow this spec's expansion order, and are ``None`` when
# that expansion is a regex over bodies whose destination paths are not yet known.
_SiteSpec = tuple[
    tuple[str, ...] | None,
    wp.transform,
    tuple[float, float, float],
    bool,
    tuple[int, ...] | None,
    tuple[str, ...] | None,
]


def _destination_prim_paths(
    prim_path: str, source_root: str | None, destination_template: str | None, env_ids: tuple[int, ...] | None
) -> tuple[str, ...]:
    """Map a clone source prim to its per-environment destination paths, or to itself when it is not
    part of a clone plan row."""
    if source_root is None or destination_template is None or env_ids is None:
        return (prim_path,)
    suffix = prim_path if source_root == "/" else prim_path[len(source_root) :]
    return tuple(destination_template.format(env_id) + suffix for env_id in env_ids)


@wp.kernel
def _compute_site_world_transforms(
    body_q: wp.array(dtype=wp.transformf),
    site_body: wp.array(dtype=wp.int32),
    site_local: wp.array(dtype=wp.transformf),
    indices: wp.array(dtype=wp.int32),
    out_pos: wp.array(dtype=wp.vec3f),
    out_quat: wp.array(dtype=wp.vec4f),
):
    """Compute world-space transforms for selected sites."""
    i = wp.tid()
    si = indices[i]
    bid = site_body[si]
    if bid == WORLD_BODY_INDEX:
        world = site_local[si]
    else:
        world = wp.transform_multiply(body_q[bid], site_local[si])
    out_pos[i] = wp.transform_get_translation(world)
    q = wp.transform_get_rotation(world)
    out_quat[i] = wp.vec4f(q[0], q[1], q[2], q[3])


@wp.kernel
def _gather_site_local_transforms(
    site_local: wp.array(dtype=wp.transformf),
    indices: wp.array(dtype=wp.int32),
    out_pos: wp.array(dtype=wp.vec3f),
    out_quat: wp.array(dtype=wp.vec4f),
):
    """Gather local transforms for selected sites."""
    i = wp.tid()
    si = indices[i]
    local_tf = site_local[si]
    out_pos[i] = wp.transform_get_translation(local_tf)
    q = wp.transform_get_rotation(local_tf)
    out_quat[i] = wp.vec4f(q[0], q[1], q[2], q[3])


@wp.kernel
def _write_site_local_from_world_poses(
    body_q: wp.array(dtype=wp.transformf),
    site_body: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.int32),
    world_pos: wp.array(dtype=wp.vec3f),
    world_quat: wp.array(dtype=wp.vec4f),
    site_local: wp.array(dtype=wp.transformf),
):
    """Update local offsets so selected sites reach desired world poses."""
    i = wp.tid()
    si = indices[i]
    w_pos = world_pos[i]
    w_q = world_quat[i]
    desired_world = wp.transform(w_pos, wp.quatf(w_q[0], w_q[1], w_q[2], w_q[3]))

    bid = site_body[si]
    if bid == WORLD_BODY_INDEX:
        site_local[si] = desired_world
    else:
        site_local[si] = wp.transform_multiply(wp.transform_inverse(body_q[bid]), desired_world)


@wp.kernel
def _write_site_local_from_local_poses(
    indices: wp.array(dtype=wp.int32),
    local_pos: wp.array(dtype=wp.vec3f),
    local_quat: wp.array(dtype=wp.vec4f),
    site_local: wp.array(dtype=wp.transformf),
):
    """Update local offsets for selected sites."""
    i = wp.tid()
    si = indices[i]
    l_pos = local_pos[i]
    l_q = local_quat[i]
    site_local[si] = wp.transform(l_pos, wp.quatf(l_q[0], l_q[1], l_q[2], l_q[3]))


@wp.kernel(enable_backward=False)
def _gather_mirrored_site_poses(
    site_positions: wp.array(dtype=wp.vec3f),
    site_orientations: wp.array(dtype=wp.vec4f),
    site_indices: wp.array(dtype=wp.int32),
    out_positions: wp.array(dtype=wp.float32, ndim=2),
    out_orientations: wp.array(dtype=wp.float32, ndim=2),
):
    """Gather the mirrored sites' world poses into the flat layout the Fabric kernels expect."""
    i = wp.tid()
    site = site_indices[i]
    position = site_positions[site]
    orientation = site_orientations[site]
    for axis in range(3):
        out_positions[i, axis] = position[axis]
    for component in range(4):
        out_orientations[i, component] = orientation[component]


@wp.kernel
def _gather_shape_scales(
    shape_scale: wp.array(dtype=wp.vec3f),
    shape_body: wp.array(dtype=wp.int32),
    site_body: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.int32),
    num_shapes: wp.int32,
    out_scales: wp.array(dtype=wp.vec3f),
):
    """Gather legacy per-site geometry scales from collision shapes on the same body."""
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
def _scatter_shape_scales(
    site_body: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.int32),
    new_scales: wp.array(dtype=wp.vec3f),
    shape_body: wp.array(dtype=wp.int32),
    num_shapes: wp.int32,
    shape_scale: wp.array(dtype=wp.vec3f),
):
    """Scatter legacy per-site geometry scales to collision shapes on the same body."""
    i = wp.tid()
    si = indices[i]
    bid = site_body[si]
    for s in range(num_shapes):
        if shape_body[s] == bid:
            shape_scale[s] = new_scales[i]


@wp.kernel
def _gather_xform_scales(
    site_xform_scale: wp.array(dtype=wp.vec3f),
    indices: wp.array(dtype=wp.int32),
    out_scales: wp.array(dtype=wp.vec3f),
):
    """Gather per-site xform scales."""
    i = wp.tid()
    out_scales[i] = site_xform_scale[indices[i]]


@wp.kernel
def _scatter_xform_scales(
    indices: wp.array(dtype=wp.int32),
    new_scales: wp.array(dtype=wp.vec3f),
    site_xform_scale: wp.array(dtype=wp.vec3f),
):
    """Scatter per-site xform scales."""
    i = wp.tid()
    site_xform_scale[indices[i]] = new_scales[i]


class NewtonSiteFrameView(BaseFrameView):
    """Batched Newton site view for non-physics frames.

    The public construction contract matches the generic :class:`FrameView`:
    callers provide a prim expression and the backend resolves the source prim
    into Newton body-local or world-local sites.
    """

    def __init__(
        self,
        prim_path: str | list[str],
        device: str = "cpu",
        validate_xform_ops: bool = True,
        stage: object | None = None,
        **kwargs,
    ):
        """Initialize the Newton site frame view.

        Args:
            prim_path: User-facing frame path pattern, or list of patterns.
            device: Warp device for GPU arrays.
            validate_xform_ops: Whether to validate source USD xform ops.
            stage: USD stage that contains the source prims.
            **kwargs: Unused.
        """
        del kwargs

        self._prim_paths = [prim_path] if isinstance(prim_path, str) else list(prim_path)
        self._prim_path = prim_path if isinstance(prim_path, str) else ", ".join(self._prim_paths)
        self._device = device
        self._prims = []

        stage = sim_utils.get_current_stage() if stage is None else stage
        self._site_specs = self._resolve_site_specs(stage, validate_xform_ops)
        self._site_labels: list[str] = []
        self._site_label_scales: list[tuple[float, float, float]] = []
        # Destination prim paths per label, in expansion order; ``None`` when that is not yet known.
        self._site_label_prim_paths: list[tuple[str, ...] | None] = []
        self._site_prim_paths: list[str] | None = None
        # Fabric mirror state, built on the first write (see :meth:`_mirror_to_fabric`).
        self._fabric_sel: FabricXformSelection | None = None
        self._mirror_disabled = False
        # Set only on the pre-model path below; released again by :meth:`close`.
        self._physics_ready_handle = None
        self._site_body: wp.array | None = None
        self._site_local: wp.array | None = None
        self._site_xform_scale: wp.array | None = None
        self._site_indices: wp.array | None = None
        self._pos_buf: wp.array | None = None
        self._quat_buf: wp.array | None = None
        self._local_pos_buf: wp.array | None = None
        self._local_quat_buf: wp.array | None = None
        self._scale_buf: wp.array | None = None
        self._pos_ta: ProxyArray | None = None
        self._quat_ta: ProxyArray | None = None
        self._local_pos_ta: ProxyArray | None = None
        self._local_quat_ta: ProxyArray | None = None
        self._scale_ta: ProxyArray | None = None
        self._count = 0

        model = NewtonManager.get_model()
        if model is not None:
            self._initialize_from_specs(model)
        else:
            for body_patterns, xform, scale, per_world, _env_ids, spec_paths in self._site_specs:
                if body_patterns is None:
                    self._site_labels.append(NewtonManager.cl_register_site(None, xform, per_world=per_world))
                    self._site_label_scales.append(scale)
                    self._site_label_prim_paths.append(spec_paths)
                else:
                    for body_pattern in body_patterns:
                        self._site_labels.append(NewtonManager.cl_register_site(body_pattern, xform))
                        self._site_label_scales.append(scale)
                        self._site_label_prim_paths.append(spec_paths)
            self._physics_ready_handle = NewtonManager.register_callback(
                self._on_physics_ready, PhysicsEvent.PHYSICS_READY, name=f"site_view_{self._prim_path}"
            )

    def _resolve_site_specs(self, stage, validate_xform_ops: bool) -> list[_SiteSpec]:
        """Resolve source prims into Newton site registration specs."""
        plan = sim_utils.SimulationContext.instance().get_clone_plan()
        model = NewtonManager.get_model()
        body_labels = list(model.body_label) if model is not None else ()
        shape_labels = list(model.shape_label) if model is not None else ()
        shape_flags = None
        use_clone_body_pattern = model is None
        specs: list[_SiteSpec] = []

        for path_expr in self._prim_paths:
            if resolve_matching_names(path_expr, body_labels, raise_when_no_match=False)[1]:
                raise ValueError(
                    f"FrameView prim '{path_expr}' is a Newton physics body. "
                    "FrameView should only be used for non-physics frames."
                )
            shape_indices, _ = resolve_matching_names(path_expr, shape_labels, raise_when_no_match=False)
            if shape_indices:
                if shape_flags is None:
                    shape_flags = model.shape_flags.numpy()
                if any(
                    int(shape_flags[index]) & int(ShapeFlags.COLLIDE_SHAPES | ShapeFlags.COLLIDE_PARTICLES)
                    for index in shape_indices
                ):
                    raise ValueError(
                        f"FrameView prim '{path_expr}' matches a Newton collision shape. "
                        "FrameView should only be used for non-physics frames."
                    )
            matches = tuple(cloner.query.iter_sources(plan, path_expr)) if plan is not None else ()
            if matches:
                for source_root, destination_template, source_path, env_ids in matches:
                    source_pattern = re.compile(source_path)
                    source_prims = sim_utils.get_all_matching_child_prims(
                        source_root,
                        lambda prim: source_pattern.fullmatch(prim.GetPath().pathString) is not None,
                        stage=stage,
                    )
                    if not source_prims:
                        raise RuntimeError(f"FrameView '{path_expr}' could not resolve source prim '{source_path}'.")
                    specs.extend(
                        self._resolve_source_prim(
                            source_prim,
                            validate_xform_ops,
                            source_root,
                            destination_template,
                            env_ids,
                            use_clone_body_pattern,
                            stage,
                        )
                        for source_prim in source_prims
                    )
                continue

            prims = sim_utils.find_matching_prims(path_expr, stage)
            if not prims:
                raise RuntimeError(f"FrameView '{path_expr}' could not resolve a source prim.")
            specs.extend(
                self._resolve_source_prim(prim, validate_xform_ops, None, None, None, use_clone_body_pattern, stage)
                for prim in prims
            )

        return specs

    def _resolve_source_prim(
        self,
        prim,
        validate_xform_ops: bool,
        source_root: str | None,
        destination_template: str | None,
        env_ids: tuple[int, ...] | None,
        use_clone_body_pattern: bool,
        stage,
    ) -> _SiteSpec:
        """Resolve one source prim into body patterns, local frame, xform scale, and destination paths."""
        prim_path = prim.GetPath().pathString
        dest_paths = _destination_prim_paths(prim_path, source_root, destination_template, env_ids)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError(
                f"FrameView prim '{prim_path}' is a Newton collision shape. "
                "FrameView should only be used for non-physics frames."
            )
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            raise ValueError(
                f"FrameView prim '{prim_path}' is a Newton physics body. "
                "FrameView should only be used for non-physics frames."
            )
        if validate_xform_ops:
            sim_utils.standardize_xform_ops(prim)
            if not sim_utils.validate_standard_xform_ops(prim):
                raise ValueError(f"FrameView prim '{prim_path}' does not have standard xform ops.")

        scale_attr = prim.GetAttribute("xformOp:scale")
        scale = (
            tuple(float(v) for v in scale_attr.Get())
            if scale_attr and scale_attr.HasAuthoredValue()
            else (1.0, 1.0, 1.0)
        )

        body_prim = prim.GetParent()
        while body_prim and body_prim.IsValid():
            if body_prim.HasAPI(UsdPhysics.RigidBodyAPI) or body_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                pos, quat = sim_utils.resolve_prim_pose(prim, body_prim)
                body_path = body_prim.GetPath().pathString
                if source_root is not None and destination_template is not None:
                    assert env_ids is not None
                    if body_path == source_root:
                        suffix = ""
                    elif body_path.startswith(source_root + "/"):
                        suffix = body_path[len(source_root) :]
                    elif source_root.startswith(body_path + "/"):
                        suffix = source_root[len(body_path) :]
                        if use_clone_body_pattern:
                            destination_root = destination_template.format(".*")
                            if not destination_root.endswith(suffix):
                                raise RuntimeError(
                                    f"FrameView destination root '{destination_root}' does not end with '{suffix}'."
                                )
                            return (
                                (destination_root[: -len(suffix)],),
                                wp.transform(pos, quat),
                                scale,
                                False,
                                env_ids,
                                dest_paths,
                            )
                        body_patterns = []
                        for env_id in env_ids:
                            destination_root = destination_template.format(env_id)
                            if not destination_root.endswith(suffix):
                                raise RuntimeError(
                                    f"FrameView destination root '{destination_root}' does not end with '{suffix}'."
                                )
                            body_patterns.append(destination_root[: -len(suffix)])
                        return tuple(body_patterns), wp.transform(pos, quat), scale, False, env_ids, dest_paths
                    else:
                        raise RuntimeError(f"FrameView source body '{body_path}' is not under '{source_root}'.")
                    if use_clone_body_pattern:
                        body_patterns = (destination_template.format(".*") + suffix,)
                    else:
                        body_patterns = tuple(destination_template.format(env_id) + suffix for env_id in env_ids)
                else:
                    body_patterns = (body_path,)
                return body_patterns, wp.transform(pos, quat), scale, False, env_ids, dest_paths
            body_prim = body_prim.GetParent()

        ref_path = source_root
        if source_root is not None and destination_template is not None:
            template_prefix, _ = cloner.path.split(destination_template)
            source_suffix = cloner.path.relativize(source_root, template_prefix + "{}")
            if source_suffix is not None:
                ref_path = source_root[: -len(source_suffix)] if source_suffix else source_root
        ref_prim = stage.GetPrimAtPath(ref_path) if ref_path is not None else None
        pos, quat = sim_utils.resolve_prim_pose(prim, ref_prim if ref_prim and ref_prim.IsValid() else None)
        return None, wp.transform(pos, quat), scale, source_root is not None, env_ids, dest_paths

    def _on_physics_ready(self, _event) -> None:
        """Callback invoked when the Newton model becomes available."""
        self._initialize_from_site_map(NewtonManager.get_model())

    def _initialize_from_site_map(self, model) -> None:
        """Initialize arrays from injected Newton sites."""
        site_map = NewtonManager._cl_site_index_map
        body_t = wp.to_torch(model.shape_body)
        xform_t = wp.to_torch(model.shape_transform)
        site_bodies: list[int] = []
        site_locals: list[list[float]] = []
        site_scales: list[tuple[float, float, float]] = []
        site_prim_paths: list[str] | None = []

        for site_label, scale, label_paths in zip(
            self._site_labels, self._site_label_scales, self._site_label_prim_paths, strict=True
        ):
            global_idx, per_world = site_map[site_label]
            site_indices = (
                [global_idx] if per_world is None else [site_idx for sites in per_world for site_idx in sites]
            )
            for site_idx in site_indices:
                site_bodies.append(int(body_t[site_idx].item()))
                site_locals.append([float(v) for v in xform_t[site_idx].tolist()])
                site_scales.append(scale)
            if site_prim_paths is not None:
                if label_paths is None or len(label_paths) != len(site_indices):
                    site_prim_paths = None
                else:
                    site_prim_paths.extend(label_paths)

        self._create_buffers(site_bodies, site_locals, site_scales, site_prim_paths)

    def _initialize_from_specs(self, model) -> None:
        """Initialize arrays directly from resolved specs and Newton body labels."""
        body_labels = list(model.body_label)
        # Exact label -> index map, built once. Replicated frames expand to one concrete
        # body path per environment, so matching each against every label via regex is
        # ``O(num_envs * num_bodies)`` (quadratic in ``num_envs``). Fast-pathing literal
        # paths through this map keeps the common per-environment case linear; genuine
        # regex patterns (e.g. the cloned ``.*`` pattern) still fall back to a full scan.
        label_to_index = {label: idx for idx, label in enumerate(body_labels)}
        site_bodies: list[int] = []
        site_locals: list[list[float]] = []
        site_scales: list[tuple[float, float, float]] = []

        site_prim_paths: list[str] | None = []

        def record_paths(spec_paths: tuple[str, ...] | None, expected: int) -> None:
            """Append this spec's destination paths, or give up on mirroring if they do not line up."""
            nonlocal site_prim_paths
            if site_prim_paths is None:
                return
            if spec_paths is None or len(spec_paths) != expected:
                site_prim_paths = None
            else:
                site_prim_paths.extend(spec_paths)

        for body_patterns, xform, scale, per_world, env_ids, spec_paths in self._site_specs:
            if body_patterns is None:
                if per_world:
                    if NewtonManager._world_xforms is None:
                        raise RuntimeError(f"FrameView '{self._prim_path}' needs Newton cloned-world transforms.")
                    world_ids = range(len(NewtonManager._world_xforms)) if env_ids is None else env_ids
                    for world_id in world_ids:
                        world_xform = NewtonManager._world_xforms[world_id]
                        site_bodies.append(WORLD_BODY_INDEX)
                        site_locals.append([float(v) for v in wp.transform_multiply(world_xform, xform)])
                        site_scales.append(scale)
                    record_paths(spec_paths, len(world_ids))
                else:
                    site_bodies.append(WORLD_BODY_INDEX)
                    site_locals.append([float(v) for v in xform])
                    site_scales.append(scale)
                    record_paths(spec_paths, 1)
                continue

            for index, body_pattern in enumerate(body_patterns):
                exact_index = label_to_index.get(body_pattern) if not _has_regex_tokens(body_pattern) else None
                if exact_index is not None:
                    matched_indices = [exact_index]
                else:
                    matched_indices, _ = resolve_matching_names(body_pattern, body_labels, raise_when_no_match=False)
                if not matched_indices:
                    raise ValueError(
                        f"FrameView '{self._prim_path}' body pattern '{body_pattern}' matched no Newton bodies."
                    )

                for body_idx in matched_indices:
                    site_bodies.append(body_idx)
                    site_locals.append([float(v) for v in xform])
                    site_scales.append(scale)
                record_paths(None if spec_paths is None else (spec_paths[index],), len(matched_indices))

        self._create_buffers(site_bodies, site_locals, site_scales, site_prim_paths)

    def _create_buffers(
        self,
        site_bodies: list[int],
        site_locals: list[list[float]],
        site_scales: list[tuple[float, float, float]],
        site_prim_paths: list[str] | None = None,
    ) -> None:
        """Allocate view buffers from body indices, local transforms, and destination prim paths."""
        self._count = len(site_bodies)
        self._site_prim_paths = (
            site_prim_paths if site_prim_paths is not None and len(site_prim_paths) == self._count else None
        )
        if self._site_prim_paths is None and self._count:
            logger.warning(
                f"FrameView '{self._prim_path}' could not pair its sites with destination prims; pose writes"
                " will update Newton state but will not be visible to the renderer."
            )
        device = self._device
        self._site_body = wp.array(site_bodies, dtype=wp.int32, device=device)
        self._site_local = wp.array([wp.transform(*x) for x in site_locals], dtype=wp.transformf, device=device)
        self._site_xform_scale = wp.array([wp.vec3f(*scale) for scale in site_scales], dtype=wp.vec3f, device=device)
        self._site_indices = wp.array(list(range(self._count)), dtype=wp.int32, device=device)
        self._pos_buf = wp.zeros(self._count, dtype=wp.vec3f, device=device)
        self._quat_buf = wp.zeros(self._count, dtype=wp.vec4f, device=device)
        self._local_pos_buf = wp.zeros(self._count, dtype=wp.vec3f, device=device)
        self._local_quat_buf = wp.zeros(self._count, dtype=wp.vec4f, device=device)
        self._scale_buf = wp.zeros(self._count, dtype=wp.vec3f, device=device)
        self._pos_ta = ProxyArray(self._pos_buf)
        self._quat_ta = ProxyArray(self._quat_buf)
        self._local_pos_ta = ProxyArray(self._local_pos_buf)
        self._local_quat_ta = ProxyArray(self._local_quat_buf)
        self._scale_ta = ProxyArray(self._site_xform_scale)

    def _mirror_to_fabric(self) -> None:
        """Stamp the current site world poses onto the Fabric transforms the renderer reads.

        Called when a transform writer scope exits; Newton keeps site poses in Warp state the RTX
        renderer never sees. The local matrix is written too because Newton's body sync ends in a
        Fabric hierarchy pass that forward-propagates ``parent * local``, which would otherwise
        overwrite the world matrix for frames parented under a body. Scale is left untouched: the
        empty scale input makes the composition kernel preserve the matrix's existing (accumulated
        parent) scale, matching how the body sync treats authored USD scale.
        """
        if self._fabric_sel is None and not self._initialize_fabric_mirror():
            return

        # Bodies sync to Fabric only at render cadence, so after a ``render=False`` step the local
        # derivation below would read a parent still at its last rendered pose. No-op when clean.
        NewtonManager.sync_transforms_to_usd()

        count = self._fabric_sel.count
        pos_ta, quat_ta = self._get_world_poses_impl(None)
        wp.launch(
            _gather_mirrored_site_poses,
            dim=count,
            inputs=[pos_ta.warp, quat_ta.warp, self._mirror_site_indices],
            outputs=[self._mirror_positions, self._mirror_orientations],
            device=self._device,
        )
        world_ifa, local_ifa = self._fabric_sel.child_ifas()
        wp.launch(
            fabric_utils.compose_indexed_fabric_transforms,
            dim=count,
            inputs=[
                world_ifa,
                self._mirror_positions,
                self._mirror_orientations,
                self._mirror_empty_scales,
                False,
                False,
                False,
                self._fabric_sel.view_indices,
            ],
            device=self._device,
        )
        wp.launch(
            fabric_utils.update_indexed_local_matrix_from_world,
            dim=count,
            inputs=[world_ifa, self._fabric_sel.parent_world_ifa(), local_ifa, self._fabric_sel.view_indices],
            device=self._device,
        )

    def _initialize_fabric_mirror(self) -> bool:
        """Build the Fabric selection backing :meth:`_mirror_to_fabric`, returning whether there is
        anything to mirror (a ``False`` result is sticky).

        Seeding from USD is off because Newton never writes poses back, so it would reset the prim to
        its spawn pose. Coverage can be partial: Newton clones physics without cloning USD, so a site
        can be real while its destination prim exists on no stage, and those have nothing to draw.
        """
        if self._mirror_disabled or self._site_prim_paths is None or self._count == 0:
            self._mirror_disabled = True
            return False
        try:
            selection = FabricXformSelection(
                self._site_prim_paths,
                self._device,
                owner=type(self).__name__,
                seed_from_usd=False,
                skip_missing_prims=True,
            )
        except ImportError:
            # No Fabric runtime (kitless run): the site state is still correct, nothing consumes it.
            self._mirror_disabled = True
            logger.info("Fabric runtime unavailable; Newton site poses will not be mirrored to prims.")
            return False

        count = selection.count
        if count == 0:
            self._mirror_disabled = True
            return False
        if count != len(self._site_prim_paths):
            logger.info(
                f"FrameView '{self._prim_path}' mirrors {count} of {len(self._site_prim_paths)} sites to Fabric;"
                " the rest have no prim on the stage (physics-only clones) and nothing to render."
            )

        # Refreshing the read-write selection is what notifies the renderer: a write through the
        # read-only one lands in Fabric, but the image keeps showing the old pose.
        selection.read_write = True
        self._fabric_sel = selection
        self._mirror_site_indices = wp.array(selection.kept_indices, dtype=wp.int32, device=self._device)
        self._mirror_positions = wp.empty((count, 3), dtype=wp.float32, device=self._device)
        self._mirror_orientations = wp.empty((count, 4), dtype=wp.float32, device=self._device)
        self._mirror_empty_scales = wp.zeros((0, 0), dtype=wp.float32, device=self._device)
        return True

    def close(self) -> None:
        """Release the Fabric attributes and model-ready callback owned by this view.

        The view must not be used afterwards. Calling :meth:`close` again is a no-op. If
        :meth:`close` is never called, the same cleanup runs best-effort from ``__del__`` --
        collection timing is up to the interpreter, so only :meth:`close` is deterministic.
        """
        handle = self._physics_ready_handle
        self._physics_ready_handle = None  # cleared first so a repeat close() cannot deregister twice
        if handle is not None:
            handle.deregister()
        if self._fabric_sel is not None:
            self._fabric_sel.close()
            self._fabric_sel = None
        self._mirror_disabled = True

    def __del__(self, _sys=sys):
        """Best-effort cleanup when the view is collected without :meth:`close`.

        Follows the repo's shutdown-safe ``__del__`` idiom (see
        :meth:`~isaaclab.envs.ManagerBasedEnv.__del__`): ``sys`` is bound as a default argument so it
        survives module teardown, and nothing runs during interpreter finalization, when calling into
        Kit can crash and the Fabric tags die with Fabric anyway.
        """
        if _sys.is_finalizing() or _sys.meta_path is None:
            return
        with contextlib.suppress(Exception):  # never propagate from __del__
            self.close()

    @property
    def prims(self) -> list:
        """List of USD prims being managed by this view.

        Newton site views do not retain USD prim handles.
        """
        return self._prims

    @property
    def count(self) -> int:
        """Number of frames in this view."""
        return self._count

    @property
    def device(self) -> str:
        """Device where arrays are allocated."""
        return self._device

    # ------------------------------------------------------------------
    # Writer factory hooks (pass-through; Newton has no separate Fabric storage)
    # ------------------------------------------------------------------

    def _make_world_space_writer(self) -> FrameViewWorldSpaceWriter:
        return _NewtonWorldSpaceWriter(self)

    def _make_local_space_writer(self) -> FrameViewLocalSpaceWriter:
        return _NewtonLocalSpaceWriter(self)

    # ------------------------------------------------------------------
    # Backend hooks
    # ------------------------------------------------------------------

    def _get_world_poses_impl(self, indices: wp.array | None = None) -> tuple[ProxyArray, ProxyArray]:
        """Get world-space positions and orientations."""
        state = NewtonManager.get_state_0()
        site_indices = self._site_indices if indices is None else indices
        n = self.count if indices is None else len(indices)
        pos_buf = self._pos_buf if indices is None else wp.zeros(n, dtype=wp.vec3f, device=self._device)
        quat_buf = self._quat_buf if indices is None else wp.zeros(n, dtype=wp.vec4f, device=self._device)

        wp.launch(
            _compute_site_world_transforms,
            dim=n,
            inputs=[state.body_q, self._site_body, self._site_local, site_indices],
            outputs=[pos_buf, quat_buf],
            device=self._device,
        )
        if indices is None:
            return self._pos_ta, self._quat_ta
        return ProxyArray(pos_buf), ProxyArray(quat_buf)

    def _apply_world_pose_write(
        self,
        positions: wp.array | None = None,
        orientations: wp.array | None = None,
        indices: wp.array | None = None,
    ) -> None:
        """Set world-space positions and/or orientations."""
        if positions is None and orientations is None:
            return

        state = NewtonManager.get_state_0()
        if positions is None or orientations is None:
            cur_pos_ta, cur_quat_ta = self._get_world_poses_impl(indices)
            if positions is None:
                positions = cur_pos_ta.warp
            if orientations is None:
                orientations = cur_quat_ta.warp

        site_indices = self._site_indices if indices is None else indices
        n = self.count if indices is None else len(indices)
        wp.launch(
            _write_site_local_from_world_poses,
            dim=n,
            inputs=[state.body_q, self._site_body, site_indices, positions, orientations, self._site_local],
            device=self._device,
        )

    def _get_local_poses_impl(self, indices: wp.array | None = None) -> tuple[ProxyArray, ProxyArray]:
        """Get body-local positions and orientations."""
        site_indices = self._site_indices if indices is None else indices
        n = self.count if indices is None else len(indices)
        pos_buf = self._local_pos_buf if indices is None else wp.zeros(n, dtype=wp.vec3f, device=self._device)
        quat_buf = self._local_quat_buf if indices is None else wp.zeros(n, dtype=wp.vec4f, device=self._device)

        wp.launch(
            _gather_site_local_transforms,
            dim=n,
            inputs=[self._site_local, site_indices],
            outputs=[pos_buf, quat_buf],
            device=self._device,
        )
        if indices is None:
            return self._local_pos_ta, self._local_quat_ta
        return ProxyArray(pos_buf), ProxyArray(quat_buf)

    def _apply_local_pose_write(
        self,
        translations: wp.array | None = None,
        orientations: wp.array | None = None,
        indices: wp.array | None = None,
    ) -> None:
        """Set body-local translations and/or orientations."""
        if translations is None and orientations is None:
            return

        if translations is None or orientations is None:
            cur_pos_ta, cur_quat_ta = self._get_local_poses_impl(indices)
            if translations is None:
                translations = cur_pos_ta.warp
            if orientations is None:
                orientations = cur_quat_ta.warp

        site_indices = self._site_indices if indices is None else indices
        n = self.count if indices is None else len(indices)
        wp.launch(
            _write_site_local_from_local_poses,
            dim=n,
            inputs=[site_indices, translations, orientations, self._site_local],
            device=self._device,
        )

    # ------------------------------------------------------------------
    # Scales
    # ------------------------------------------------------------------

    def _get_world_scales_impl(self, indices: wp.array | None = None) -> ProxyArray:
        """Get per-site world xform scales.

        These are transform scales, matching the USD FrameView scale API.  They
        are intentionally separate from Newton collision shape geometry sizes.
        """
        if indices is None:
            return self._scale_ta
        n = len(indices)
        out = wp.zeros(n, dtype=wp.vec3f, device=self._device)
        wp.launch(
            _gather_xform_scales,
            dim=n,
            inputs=[self._site_xform_scale, indices],
            outputs=[out],
            device=self._device,
        )
        return ProxyArray(out)

    def _get_local_scales_impl(self, indices: wp.array | None = None) -> ProxyArray:
        """Get per-site local xform scales.

        These are transform scales, matching the USD FrameView scale API.  They
        are intentionally separate from Newton collision shape geometry sizes.
        """
        return self._get_world_scales_impl(indices)

    def _apply_world_scale_write(self, scales: wp.array, indices: wp.array | None = None) -> None:
        """Set per-site world xform scales.

        These update transform scale state only; use deprecated ``set_scales`` if
        legacy Newton collision shape geometry-scale behavior is required.
        """
        if indices is None:
            indices = self._site_indices
        n = self.count if indices is self._site_indices else len(indices)
        wp.launch(
            _scatter_xform_scales,
            dim=n,
            inputs=[indices, scales, self._site_xform_scale],
            device=self._device,
        )

    def _apply_local_scale_write(self, scales: wp.array, indices: wp.array | None = None) -> None:
        """Set per-site local xform scales.

        These update transform scale state only; use deprecated ``set_scales`` if
        legacy Newton collision shape geometry-scale behavior is required.
        """
        self._apply_world_scale_write(scales, indices)

    def _get_legacy_shape_scales(self, indices: wp.array | None = None) -> ProxyArray:
        """Get Newton legacy geometry scales from collision shapes."""
        model = NewtonManager.get_model()
        num_shapes = model.shape_count
        site_indices = self._site_indices if indices is None else indices
        n = self.count if indices is None else len(indices)
        out = wp.zeros(n, dtype=wp.vec3f, device=self._device)
        wp.launch(
            _gather_shape_scales,
            dim=n,
            inputs=[model.shape_scale, model.shape_body, self._site_body, site_indices, num_shapes],
            outputs=[out],
            device=self._device,
        )
        return ProxyArray(out)

    def _set_legacy_shape_scales(self, scales: wp.array, indices: wp.array | None = None) -> None:
        """Set Newton legacy geometry scales on collision shapes."""
        model = NewtonManager.get_model()
        num_shapes = model.shape_count
        site_indices = self._site_indices if indices is None else indices
        n = self.count if indices is None else len(indices)
        wp.launch(
            _scatter_shape_scales,
            dim=n,
            inputs=[self._site_body, site_indices, scales, model.shape_body, num_shapes, model.shape_scale],
            device=self._device,
        )

    def _get_scales_impl(self, indices: wp.array | None = None) -> ProxyArray:
        """Newton legacy: get_scales returns collision shape geometry scales."""
        return self._get_legacy_shape_scales(indices)

    def _set_scales_impl(self, scales: wp.array, indices: wp.array | None = None) -> None:
        """Newton legacy: deprecated set_scales writes collision shape geometry scales.

        Newton's legacy ``set_scales`` path is *not* routed through the
        :class:`FrameViewSpaceWriterBase` API because it targets a different state
        (collision-shape geometry sizes) than the transform-scale state that
        the writer's :meth:`~FrameViewSpaceWriterBase.set_scales` operates on.
        """
        self._set_legacy_shape_scales(scales, indices)


# ----------------------------------------------------------------------
# Pass-through writer classes
# ----------------------------------------------------------------------


class _NewtonWriterMixin:
    """Mirrors the scope's pose writes onto Fabric on exit.

    The mirror is a full-view synchronization, so it runs only when the scope actually moved a site:
    a getter-only scope, ``set_poses(None, None)``, and a scale-only scope all leave the mirrored
    poses untouched (scale is deliberately not mirrored -- see :meth:`NewtonSiteFrameView._mirror_to_fabric`).

    **Exception safety.** The mirror also runs while an exception unwinds, because the Newton-side
    write is already committed: skipping it would strand the rendered prim at a stale pose until the
    next successful write. If the mirror itself fails during unwinding (typically because the
    original exception poisoned the CUDA stream) the failure is logged and the original exception
    propagates -- masking it would hide the actual root cause.
    """

    def _enter_impl(self) -> None:
        self._wrote_poses = False

    def _exit_impl(self, exc_type, exc_val, exc_tb) -> None:
        if not self._wrote_poses:
            return
        try:
            self._view._mirror_to_fabric()  # type: ignore[attr-defined]
        except Exception as mirror_exc:  # noqa: BLE001 -- see the exception-safety note above
            if exc_type is None:
                raise
            logger.error(
                "Newton frame-view writer scope: best-effort Fabric mirror failed during exception "
                "handling: %s. The rendered prim keeps its previous pose until the next pose write.",
                mirror_exc,
            )


class _NewtonWorldSpaceWriter(_NewtonWriterMixin, FrameViewWorldSpaceWriter):
    """Newton world-space writer: pass-through to backend ``_apply_*`` hooks."""

    def set_poses(self, positions=None, orientations=None, indices=None) -> None:
        if positions is None and orientations is None:
            return
        self._wrote_poses = True
        self._view._apply_world_pose_write(positions, orientations, indices)  # type: ignore[attr-defined]

    def set_scales(self, scales, indices=None) -> None:
        self._view._apply_world_scale_write(scales, indices)  # type: ignore[attr-defined]

    def get_poses(self, indices=None) -> tuple[ProxyArray, ProxyArray]:
        return self._view._get_world_poses_impl(indices)  # type: ignore[attr-defined]

    def get_scales(self, indices=None) -> ProxyArray:
        return self._view._get_world_scales_impl(indices)  # type: ignore[attr-defined]


class _NewtonLocalSpaceWriter(_NewtonWriterMixin, FrameViewLocalSpaceWriter):
    """Newton local-space writer: pass-through to backend ``_apply_*`` hooks."""

    def set_poses(self, positions=None, orientations=None, indices=None) -> None:
        if positions is None and orientations is None:
            return
        self._wrote_poses = True
        self._view._apply_local_pose_write(positions, orientations, indices)  # type: ignore[attr-defined]

    def set_scales(self, scales, indices=None) -> None:
        self._view._apply_local_scale_write(scales, indices)  # type: ignore[attr-defined]

    def get_poses(self, indices=None) -> tuple[ProxyArray, ProxyArray]:
        return self._view._get_local_poses_impl(indices)  # type: ignore[attr-defined]

    def get_scales(self, indices=None) -> ProxyArray:
        return self._view._get_local_scales_impl(indices)  # type: ignore[attr-defined]
