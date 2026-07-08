# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-backed FrameView using Newton body labels and injected sites."""

from __future__ import annotations

import logging

import warp as wp

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.cloner.cloner_utils import get_suffix, iter_clone_plan_matches, split_clone_template
from isaaclab.physics import PhysicsEvent
from isaaclab.sim.views.base_frame_view import BaseFrameView
from isaaclab.sim.views.xform_space_writer import FrameViewLocalSpaceWriter, FrameViewWorldSpaceWriter
from isaaclab.utils.string import resolve_matching_names
from isaaclab.utils.warp import ProxyArray

from isaaclab_newton.physics.newton_manager import NewtonManager

logger = logging.getLogger(__name__)

WORLD_BODY_INDEX = -1


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
            for body_patterns, xform, scale, per_world, _env_ids in self._site_specs:
                if body_patterns is None:
                    self._site_labels.append(NewtonManager.cl_register_site(None, xform, per_world=per_world))
                    self._site_label_scales.append(scale)
                else:
                    for body_pattern in body_patterns:
                        self._site_labels.append(NewtonManager.cl_register_site(body_pattern, xform))
                        self._site_label_scales.append(scale)
            self._physics_ready_handle = NewtonManager.register_callback(
                self._on_physics_ready, PhysicsEvent.PHYSICS_READY, name=f"site_view_{self._prim_path}"
            )

    def _resolve_site_specs(
        self, stage, validate_xform_ops: bool
    ) -> list[tuple[tuple[str, ...] | None, wp.transform, tuple[float, float, float], bool, tuple[int, ...] | None]]:
        """Resolve source prims into Newton site registration specs."""
        plan = sim_utils.SimulationContext.instance().get_clone_plan()
        model = NewtonManager.get_model()
        body_labels = list(model.body_label) if model is not None else ()
        shape_labels = list(model.shape_label) if model is not None else ()
        use_clone_body_pattern = model is None
        specs: list[
            tuple[tuple[str, ...] | None, wp.transform, tuple[float, float, float], bool, tuple[int, ...] | None]
        ] = []

        for path_expr in self._prim_paths:
            if resolve_matching_names(path_expr, body_labels, raise_when_no_match=False)[1]:
                raise ValueError(
                    f"FrameView prim '{path_expr}' is a Newton physics body. "
                    "FrameView should only be used for non-physics frames."
                )
            if resolve_matching_names(path_expr, shape_labels, raise_when_no_match=False)[1]:
                raise ValueError(
                    f"FrameView prim '{path_expr}' is a Newton collision shape. "
                    "FrameView should only be used for non-physics frames."
                )
            matches = tuple(iter_clone_plan_matches(plan, path_expr)) if plan is not None else ()
            if matches:
                for source_root, destination_template, source_path, env_ids in matches:
                    source_prim = None
                    if not any(token in source_path for token in "*[]()+?|\\"):
                        source_prim = stage.GetPrimAtPath(source_path)
                    if source_prim is None or not source_prim.IsValid():
                        source_prim = sim_utils.find_first_matching_prim(source_path, stage)
                    if source_prim is None or not source_prim.IsValid():
                        raise RuntimeError(f"FrameView '{path_expr}' could not resolve source prim '{source_path}'.")
                    specs.append(
                        self._resolve_source_prim(
                            source_prim,
                            validate_xform_ops,
                            source_root,
                            destination_template,
                            env_ids,
                            use_clone_body_pattern,
                            stage,
                        )
                    )
                continue

            prim = sim_utils.find_first_matching_prim(path_expr, stage)
            if prim is None or not prim.IsValid():
                raise RuntimeError(f"FrameView '{path_expr}' could not resolve a source prim.")
            specs.append(
                self._resolve_source_prim(prim, validate_xform_ops, None, None, None, use_clone_body_pattern, stage)
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
    ) -> tuple[tuple[str, ...] | None, wp.transform, tuple[float, float, float], bool, tuple[int, ...] | None]:
        """Resolve one source prim into body patterns, local frame, and xform scale."""
        prim_path = prim.GetPath().pathString
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
                            return (destination_root[: -len(suffix)],), wp.transform(pos, quat), scale, False, env_ids
                        body_patterns = []
                        for env_id in env_ids:
                            destination_root = destination_template.format(env_id)
                            if not destination_root.endswith(suffix):
                                raise RuntimeError(
                                    f"FrameView destination root '{destination_root}' does not end with '{suffix}'."
                                )
                            body_patterns.append(destination_root[: -len(suffix)])
                        return tuple(body_patterns), wp.transform(pos, quat), scale, False, env_ids
                    else:
                        raise RuntimeError(f"FrameView source body '{body_path}' is not under '{source_root}'.")
                    if use_clone_body_pattern:
                        body_patterns = (destination_template.format(".*") + suffix,)
                    else:
                        body_patterns = tuple(destination_template.format(env_id) + suffix for env_id in env_ids)
                else:
                    body_patterns = (body_path,)
                return body_patterns, wp.transform(pos, quat), scale, False, env_ids
            body_prim = body_prim.GetParent()

        ref_path = source_root
        if source_root is not None and destination_template is not None:
            template_prefix, _ = split_clone_template(destination_template)
            source_suffix = get_suffix(source_root, template_prefix + "{}")
            if source_suffix is not None:
                ref_path = source_root[: -len(source_suffix)] if source_suffix else source_root
        ref_prim = stage.GetPrimAtPath(ref_path) if ref_path is not None else None
        pos, quat = sim_utils.resolve_prim_pose(prim, ref_prim if ref_prim and ref_prim.IsValid() else None)
        return None, wp.transform(pos, quat), scale, source_root is not None, env_ids

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

        for site_label, scale in zip(self._site_labels, self._site_label_scales, strict=True):
            global_idx, per_world = site_map[site_label]
            site_indices = (
                [global_idx] if per_world is None else [site_idx for sites in per_world for site_idx in sites]
            )
            for site_idx in site_indices:
                site_bodies.append(int(body_t[site_idx].item()))
                site_locals.append([float(v) for v in xform_t[site_idx].tolist()])
                site_scales.append(scale)

        self._create_buffers(site_bodies, site_locals, site_scales)

    def _initialize_from_specs(self, model) -> None:
        """Initialize arrays directly from resolved specs and Newton body labels."""
        body_labels = list(model.body_label)
        site_bodies: list[int] = []
        site_locals: list[list[float]] = []
        site_scales: list[tuple[float, float, float]] = []

        for body_patterns, xform, scale, per_world, env_ids in self._site_specs:
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
                else:
                    site_bodies.append(WORLD_BODY_INDEX)
                    site_locals.append([float(v) for v in xform])
                    site_scales.append(scale)
                continue

            for body_pattern in body_patterns:
                matched_indices, _ = resolve_matching_names(body_pattern, body_labels, raise_when_no_match=False)
                if not matched_indices:
                    raise ValueError(
                        f"FrameView '{self._prim_path}' body pattern '{body_pattern}' matched no Newton bodies."
                    )

                for body_idx in matched_indices:
                    site_bodies.append(body_idx)
                    site_locals.append([float(v) for v in xform])
                    site_scales.append(scale)

        self._create_buffers(site_bodies, site_locals, site_scales)

    def _create_buffers(
        self,
        site_bodies: list[int],
        site_locals: list[list[float]],
        site_scales: list[tuple[float, float, float]],
    ) -> None:
        """Allocate view buffers from body indices and local transforms."""
        self._count = len(site_bodies)
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


class _NewtonWorldSpaceWriter(FrameViewWorldSpaceWriter):
    """Newton world-space writer: pass-through to backend ``_apply_*`` hooks."""

    def set_poses(self, positions=None, orientations=None, indices=None) -> None:
        self._view._apply_world_pose_write(positions, orientations, indices)  # type: ignore[attr-defined]

    def set_scales(self, scales, indices=None) -> None:
        self._view._apply_world_scale_write(scales, indices)  # type: ignore[attr-defined]

    def get_poses(self, indices=None) -> tuple[ProxyArray, ProxyArray]:
        return self._view._get_world_poses_impl(indices)  # type: ignore[attr-defined]

    def get_scales(self, indices=None) -> ProxyArray:
        return self._view._get_world_scales_impl(indices)  # type: ignore[attr-defined]


class _NewtonLocalSpaceWriter(FrameViewLocalSpaceWriter):
    """Newton local-space writer: pass-through to backend ``_apply_*`` hooks."""

    def set_poses(self, positions=None, orientations=None, indices=None) -> None:
        self._view._apply_local_pose_write(positions, orientations, indices)  # type: ignore[attr-defined]

    def set_scales(self, scales, indices=None) -> None:
        self._view._apply_local_scale_write(scales, indices)  # type: ignore[attr-defined]

    def get_poses(self, indices=None) -> tuple[ProxyArray, ProxyArray]:
        return self._view._get_local_poses_impl(indices)  # type: ignore[attr-defined]

    def get_scales(self, indices=None) -> ProxyArray:
        return self._view._get_local_scales_impl(indices)  # type: ignore[attr-defined]
