# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX FrameView with Fabric GPU acceleration."""

from __future__ import annotations

import logging

import torch
import warp as wp

from pxr import Gf, Usd, UsdGeom

from isaaclab.app.settings_manager import SettingsManager
from isaaclab.sim.views.base_frame_view import BaseFrameView
from isaaclab.sim.views.usd_frame_view import UsdFrameView
from isaaclab.sim.views.xform_space_writer import FrameViewLocalSpaceWriter, FrameViewWorldSpaceWriter
from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.warp import fabric as fabric_utils

logger = logging.getLogger(__name__)


def _to_float32_2d(a: wp.array | torch.Tensor) -> wp.array | torch.Tensor:
    """Ensure array is compatible with Fabric kernels (2-D float32).

    For ``wp.array`` with vec dtypes (``vec3f``, ``vec4f``), uses
    :meth:`wp.array.view` for zero-copy reinterpretation.
    ``torch.Tensor`` and already-correct 2-D float32 arrays pass through.
    """
    if not isinstance(a, wp.array):
        return a
    if a.shape[0] == 0:
        return a
    if a.ndim == 2 and a.dtype == wp.float32:
        return a
    return a.view(dtype=wp.float32)


class FabricFrameView(BaseFrameView):
    """FrameView with Fabric GPU acceleration for the PhysX backend.

    Uses composition: holds a :class:`UsdFrameView` internally for USD
    fallback and non-accelerated operations (visibility, and all pose/scale
    operations when Fabric is disabled).

    When Fabric is enabled, world-pose, local-pose, and scale operations run
    on the GPU via Warp kernels that read and write
    ``omni:fabric:worldMatrix`` and ``omni:fabric:localMatrix`` directly.
    All other operations delegate to the internal USD view.

    All writes go through the writer-scope API
    (:meth:`xform_world_space_writer` / :meth:`xform_local_space_writer`,
    recommended) or the
    convenience :meth:`set_world_poses` / :meth:`set_local_poses` / etc. helpers
    inherited from :class:`BaseFrameView`.

    Behavior (Fabric path):

    * **Leaf-prim assumption.**  This view manages a flat set of sibling prims
      (e.g. all cameras under ``/World/Env_*/Camera``).  It does NOT propagate
      transforms to child prims.  If a managed prim has children whose world
      matrices depend on the parent, those children must be updated via a
      separate view, a physics step, or ``IFabricHierarchy.update_world_xforms``.
    * **No write-back to USD.**  Fabric writes update only
      ``omni:fabric:worldMatrix`` / ``omni:fabric:localMatrix``; the prim's
      USD ``xformOp:*`` attributes are unchanged.  Downstream consumers that
      read the prim's USD attributes after a Fabric write will see stale
      values until the next USD-side sync.
    * **Eager dual-write inside a writer scope (no dirty tracking).**
      When a writer scope is open, all writes go to the primary attribute
      (``worldMatrix`` for the world writer, ``localMatrix`` for the local
      writer).  On scope exit, a single Warp kernel derives the opposite
      attribute and a single ``wp.synchronize()`` runs.  After the scope
      exits, both Fabric matrices are self-consistent.  Getters launch
      their own decompose kernel and ``wp.synchronize()`` before returning,
      so a returned :class:`ProxyArray` is always immediately readable from
      either GPU or host code (no caller-side sync required).

      The opposite-space derive runs even when the scope unwinds via
      exception (including ``KeyboardInterrupt`` in interactive notebooks),
      as a best-effort to keep ``worldMatrix`` and ``localMatrix``
      mutually consistent on whatever partial-write state Fabric holds.
      The partial write itself is not rolled back -- if you need
      transactional all-or-nothing semantics, snapshot the matrices
      yourself before entering the scope.
    * **Fabric Hierarchy listeners are paused while a writer scope is
      active.**  On enter, the writer calls
      :meth:`IFabricHierarchy.track_local_xform_changes(False)` /
      :meth:`track_world_xform_changes(False)` (saving the prior state).
      Fabric itself is just a flat attribute store; the plugin that
      keeps ``omni:fabric:worldMatrix`` and ``omni:fabric:localMatrix``
      mutually consistent across the prim hierarchy is
      :class:`usdrt.hierarchy.IFabricHierarchy` (a.k.a. Fabric
      Hierarchy).  Its change tracking is pull-based: a per-attribute
      listener records writes into a private changelog, and the plugin
      drains and processes that changelog on the next call to
      ``IFabricHierarchy::update_world_xforms()`` (typically from the
      render path).  "Tracking off" just stops the listener from
      recording new entries -- writes still land in Fabric storage;
      they are simply invisible to the next ``update_world_xforms()``
      call.

      That is exactly what we want.  Inside the scope we write one
      space (world or local) and, at scope exit, derive the other in a
      single batched kernel so both matrices are mutually consistent.
      If tracking were left on, our writes would be queued in the
      changelog and the next ``update_world_xforms()`` tick would
      process them -- choosing a canonical direction (e.g. "user
      authored local, recompute world from it") and potentially
      overwriting one half of our just-consistent pair.  With tracking
      paused for the duration of the scope, the changelog stays empty
      for these prims and the next tick is a no-op for them.

      ``__exit__`` restores the prior tracking state (so we do not
      re-enable listeners the caller had previously paused).  The
      Fabric Scene Delegate (FSD) reads ``omni:fabric:worldMatrix``
      directly from Fabric storage on the render path; it observes our
      final writes unchanged.

      Note: the scope is synchronous Python code, so no simulation step
      and no render tick can run while it is open -- callers must not
      advance the simulation from inside the scope (see
      :mod:`isaaclab.sim.views.xform_space_writer` for the full contract).
      The "torn data" concern is what motivates that no-step rule; it
      is separate from why the tracking pause exists.
    * **Two persistent selections, flipped by the writer scope.**  Two
      selections are built once during ``_initialize_fabric`` and kept for
      the view's lifetime:

      .. code-block:: text

          _sel_ro :  worldMatrix=RO, localMatrix=RO   (steady state)
          _sel_rw :  worldMatrix=RW, localMatrix=RW   (inside writer scope)

      Each selection has its own bundle of indexed-fabric arrays
      (``_world_ifa_*``, ``_local_ifa_*``, ``_parent_world_ifa_*``) cached
      against the selection's path ordering.  Writer ``__enter__`` flips an
      ``_is_rw`` flag so subsequent get/set helpers resolve to the RW
      bundle; ``__exit__`` flips back to RO.  Nothing is rebuilt on the
      flip -- both bundles are always kept consistent via independent
      ``PrepareForReuse()`` polls in the accessors.

      The RO steady state tells Kit's next-tick
      ``update_world_xforms()`` that no attribute is user-authored, so it
      leaves both alone.  Combined with the tracking pause and the
      opposite-space derive at scope exit, this is what keeps the next
      render tick from overwriting our writes.
    * **Topology-adaptive.**  Fabric topology changes are detected on each
      access via per-selection ``PrepareForReuse()`` polls; the affected
      indexed arrays rebuild automatically and no manual refresh is required.

    Pose getters return :class:`~isaaclab.utils.warp.ProxyArray`; the
    convenience :meth:`set_world_poses` / :meth:`set_local_poses` helpers accept
    :class:`wp.array`.  Inside a writer scope, the writer's
    :meth:`~FrameViewSpaceWriterBase.set_poses` / :meth:`~FrameViewSpaceWriterBase.set_scales`
    accept :class:`wp.array`.
    """

    _WORLD_MATRIX_NAME = "omni:fabric:worldMatrix"
    _LOCAL_MATRIX_NAME = "omni:fabric:localMatrix"

    def __init__(
        self,
        prim_path: str,
        device: str = "cpu",
        validate_xform_ops: bool = True,
        stage: Usd.Stage | None = None,
        **kwargs,
    ):
        """Initialize the view.

        Args:
            prim_path: USD prim-path pattern to match.
            device: Device for Warp arrays. Either ``"cpu"`` or any CUDA
                device string (``"cuda:0"``, ``"cuda:1"``, ...); Fabric
                acceleration is supported on every CUDA index.
            validate_xform_ops: Whether to validate prim xform-ops.
            stage: USD stage; defaults to the current sim context's stage.
            **kwargs: Additional keyword arguments (ignored). Matches the signature of
                :class:`~isaaclab.sim.views.UsdFrameView` so that the top-level
                :class:`~isaaclab.sim.views.FrameView` factory can forward backend-agnostic
                kwargs without each backend having to know about every option.
        """
        self._usd_view = UsdFrameView(prim_path, device=device, validate_xform_ops=validate_xform_ops, stage=stage)
        self._device = device

        settings = SettingsManager.instance()
        self._use_fabric = bool(settings.get("/physics/fabricEnabled", False))

        # TODO(pv): Misleading abstraction -- FabricFrameView can fall back to USD internally;
        # the concrete class should be determined by the factory instead. (PR #5673 pv/fabric-view-no-fallback)

        self._fabric_initialized = False
        self._stage = None
        self._fabric_hierarchy = None

        # Two persistent Fabric selections.  ``_is_rw`` is True only inside
        # an active writer scope; the accessors below resolve to the matching
        # bundle of indexed arrays.
        self._sel_ro = None
        self._sel_rw = None
        self._is_rw: bool = False

        # View-side indices array (shared across both bundles).
        self._view_indices: wp.array | None = None

        # Per-selection view->fabric mappings.
        self._ro_fabric_indices: wp.array | None = None
        self._rw_fabric_indices: wp.array | None = None
        self._ro_parent_fabric_indices: wp.array | None = None
        self._rw_parent_fabric_indices: wp.array | None = None

        # Indexed fabric arrays per (selection, attribute) pair.
        self._world_ifa_ro = None
        self._local_ifa_ro = None
        self._parent_world_ifa_ro = None
        self._world_ifa_rw = None
        self._local_ifa_rw = None
        self._parent_world_ifa_rw = None

        # Sentinel passed to compose/decompose kernels for unused slots.
        self._fabric_empty_2d_array_sentinel: wp.array | None = None

    # ------------------------------------------------------------------
    # Delegated properties
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        return self._usd_view.count

    @property
    def device(self) -> str:
        """Device where arrays are allocated (cpu or cuda)."""
        return self._device

    @property
    def prims(self) -> list:
        return self._usd_view.prims

    @property
    def prim_paths(self) -> list[str]:
        return self._usd_view.prim_paths

    # ------------------------------------------------------------------
    # Delegated operations (USD-only)
    # ------------------------------------------------------------------

    def get_visibility(self, indices=None):
        return self._usd_view.get_visibility(indices)

    def set_visibility(self, visibility, indices=None):
        self._usd_view.set_visibility(visibility, indices)

    # ------------------------------------------------------------------
    # Writer factory hooks
    # ------------------------------------------------------------------

    def _make_world_space_writer(self) -> FrameViewWorldSpaceWriter:
        if not self._use_fabric:
            return _FabricFallbackWorldWriter(self)
        return _FabricWorldSpaceWriter(self)

    def _make_local_space_writer(self) -> FrameViewLocalSpaceWriter:
        if not self._use_fabric:
            return _FabricFallbackLocalWriter(self)
        return _FabricLocalSpaceWriter(self)

    # ------------------------------------------------------------------
    # Getter hooks -- read directly from Fabric (no lazy sync)
    # ------------------------------------------------------------------

    def _get_world_poses_impl(self, indices: wp.array | None = None) -> tuple[ProxyArray, ProxyArray]:
        if not self._use_fabric:
            return self._usd_view._get_world_poses_impl(indices)

        if not self._fabric_initialized:
            self._initialize_fabric()

        indices_wp = self._resolve_indices_wp(indices)
        count = indices_wp.shape[0]

        use_cached = indices is None or indices == slice(None)
        if use_cached:
            positions_wp = self._fabric_positions_buf
            orientations_wp = self._fabric_orientations_buf
        else:
            positions_wp = wp.zeros((count, 3), dtype=wp.float32, device=self._device)
            orientations_wp = wp.zeros((count, 4), dtype=wp.float32, device=self._device)

        wp.launch(
            kernel=fabric_utils.decompose_indexed_fabric_transforms,
            dim=count,
            inputs=[
                self._get_world_ifa(),
                positions_wp,
                orientations_wp,
                self._fabric_empty_2d_array_sentinel,
                indices_wp,
            ],
            device=self._device,
        )

        # Sync before returning regardless of caching path: the cached buffers
        # are reused (an in-flight kernel from a prior call must finish before
        # the new write is observable) and the fresh-allocation path must also
        # complete before the caller can read the returned ProxyArray via any
        # host-visible accessor without observing zeros.
        wp.synchronize()
        if use_cached:
            return self._fabric_positions_ta, self._fabric_orientations_ta
        return ProxyArray(positions_wp), ProxyArray(orientations_wp)

    def _get_local_poses_impl(self, indices: wp.array | None = None) -> tuple[ProxyArray, ProxyArray]:
        if not self._use_fabric:
            return self._usd_view._get_local_poses_impl(indices)

        if not self._fabric_initialized:
            self._initialize_fabric()

        indices_wp = self._resolve_indices_wp(indices)
        count = indices_wp.shape[0]

        use_cached = indices is None or indices == slice(None)
        if use_cached:
            translations_wp = self._fabric_local_translations_buf
            orientations_wp = self._fabric_local_orientations_buf
        else:
            translations_wp = wp.zeros((count, 3), dtype=wp.float32, device=self._device)
            orientations_wp = wp.zeros((count, 4), dtype=wp.float32, device=self._device)

        wp.launch(
            kernel=fabric_utils.decompose_indexed_fabric_transforms,
            dim=count,
            inputs=[
                self._get_local_ifa(),
                translations_wp,
                orientations_wp,
                self._fabric_empty_2d_array_sentinel,
                indices_wp,
            ],
            device=self._device,
        )

        # See note in _get_world_poses_impl: sync regardless of caching path.
        wp.synchronize()
        if use_cached:
            return self._fabric_local_translations_ta, self._fabric_local_orientations_ta
        return ProxyArray(translations_wp), ProxyArray(orientations_wp)

    def _get_world_scales_impl(self, indices=None) -> ProxyArray:
        if not self._use_fabric:
            return self._usd_view._get_world_scales_impl(indices)

        if not self._fabric_initialized:
            self._initialize_fabric()

        return self._decompose_scales(self._get_world_ifa(), indices)

    def _get_local_scales_impl(self, indices=None) -> ProxyArray:
        if not self._use_fabric:
            return self._usd_view._get_local_scales_impl(indices)

        if not self._fabric_initialized:
            self._initialize_fabric()

        return self._decompose_scales(self._get_local_ifa(), indices)

    def _decompose_scales(self, ro_array, indices) -> ProxyArray:
        """Shared scale-decompose path for world / local getters."""
        indices_wp = self._resolve_indices_wp(indices)
        count = indices_wp.shape[0]

        use_cached = indices is None or indices == slice(None)
        if use_cached:
            scales_wp = self._fabric_scales_buf
        else:
            scales_wp = wp.zeros((count, 3), dtype=wp.float32, device=self._device)

        wp.launch(
            kernel=fabric_utils.decompose_indexed_fabric_transforms,
            dim=count,
            inputs=[
                ro_array,
                self._fabric_empty_2d_array_sentinel,
                self._fabric_empty_2d_array_sentinel,
                scales_wp,
                indices_wp,
            ],
            device=self._device,
        )

        # See note in _get_world_poses_impl: sync regardless of caching path.
        wp.synchronize()
        if use_cached:
            return self._fabric_scales_ta
        return ProxyArray(scales_wp)

    # ------------------------------------------------------------------
    # Deprecated get_scales / set_scales hooks
    # ------------------------------------------------------------------

    def _get_scales_impl(self, indices=None) -> ProxyArray:
        """Fabric: get_scales returns world-space scales (legacy behavior)."""
        return self._get_world_scales_impl(indices)

    def _set_scales_impl(self, scales, indices=None) -> None:
        """Fabric: set_scales writes world-space scales via a one-shot writer scope."""
        with self.xform_world_space_writer() as writer:
            writer.set_scales(scales, indices)

    # ------------------------------------------------------------------
    # Internal -- helpers shared by writers + initialization
    # ------------------------------------------------------------------

    def _to_float32_2d_or_empty(self, data):
        return self._fabric_empty_2d_array_sentinel if data is None else _to_float32_2d(data)

    def _recompute_local_from_world_all(self) -> None:
        """Derive ``localMatrix = inv(parent) * worldMatrix`` for every prim in the view.

        Called from :class:`_FabricWorldSpaceWriter` ``__exit__`` to keep the
        (world, local) pair self-consistent after a world-space write.
        Storage convention: see
        :func:`isaaclab.utils.warp.fabric.update_indexed_local_matrix_from_world`.
        """
        wp.launch(
            kernel=fabric_utils.update_indexed_local_matrix_from_world,
            dim=self.count,
            inputs=[
                self._get_world_ifa(),
                self._get_parent_world_ifa(),
                self._get_local_ifa(),
                self._view_indices,
            ],
            device=self._device,
        )

    def _recompute_world_from_local_all(self) -> None:
        """Derive ``worldMatrix = parent * localMatrix`` for every prim in the view.

        Called from :class:`_FabricLocalSpaceWriter` ``__exit__`` and from
        :meth:`_sync_fabric_from_usd_initial` after seeding local matrices.
        Storage convention: see
        :func:`isaaclab.utils.warp.fabric.update_indexed_world_matrix_from_local`.
        """
        wp.launch(
            kernel=fabric_utils.update_indexed_world_matrix_from_local,
            dim=self.count,
            inputs=[
                self._get_local_ifa(),
                self._get_parent_world_ifa(),
                self._get_world_ifa(),
                self._view_indices,
            ],
            device=self._device,
        )

    # ------------------------------------------------------------------
    # Internal -- selection accessors with on-demand index rebuild
    # ------------------------------------------------------------------

    def _get_world_ifa(self):
        self._refresh_active_bundle_if_needed()
        return self._world_ifa_rw if self._is_rw else self._world_ifa_ro

    def _get_local_ifa(self):
        self._refresh_active_bundle_if_needed()
        return self._local_ifa_rw if self._is_rw else self._local_ifa_ro

    def _get_parent_world_ifa(self):
        self._refresh_active_bundle_if_needed()
        return self._parent_world_ifa_rw if self._is_rw else self._parent_world_ifa_ro

    def _refresh_active_bundle_if_needed(self) -> None:
        """Rebuild the active bundle's indexed arrays if its selection's buckets changed."""
        if self._is_rw:
            if self._world_ifa_rw is None or self._sel_rw.PrepareForReuse():
                self._rebuild_rw_arrays()
        else:
            if self._world_ifa_ro is None or self._sel_ro.PrepareForReuse():
                self._rebuild_ro_arrays()

    def _rebuild_ro_arrays(self) -> None:
        """Rebuild the four ``_sel_ro``-keyed indexed arrays (children + parents)."""
        self._ro_fabric_indices = self._compute_fabric_indices(self._sel_ro)
        self._world_ifa_ro = self._build_indexed_array(self._sel_ro, self._WORLD_MATRIX_NAME, self._ro_fabric_indices)
        self._local_ifa_ro = self._build_indexed_array(self._sel_ro, self._LOCAL_MATRIX_NAME, self._ro_fabric_indices)
        self._ro_parent_fabric_indices = self._compute_parent_fabric_indices(self._sel_ro)
        self._parent_world_ifa_ro = wp.indexedfabricarray(
            fa=wp.fabricarray(self._sel_ro, self._WORLD_MATRIX_NAME),
            indices=self._ro_parent_fabric_indices,
        )

    def _rebuild_rw_arrays(self) -> None:
        """Rebuild the four ``_sel_rw``-keyed indexed arrays (children + parents)."""
        self._rw_fabric_indices = self._compute_fabric_indices(self._sel_rw)
        self._world_ifa_rw = self._build_indexed_array(self._sel_rw, self._WORLD_MATRIX_NAME, self._rw_fabric_indices)
        self._local_ifa_rw = self._build_indexed_array(self._sel_rw, self._LOCAL_MATRIX_NAME, self._rw_fabric_indices)
        self._rw_parent_fabric_indices = self._compute_parent_fabric_indices(self._sel_rw)
        self._parent_world_ifa_rw = wp.indexedfabricarray(
            fa=wp.fabricarray(self._sel_rw, self._WORLD_MATRIX_NAME),
            indices=self._rw_parent_fabric_indices,
        )

    # ------------------------------------------------------------------
    # Internal -- index computation
    # ------------------------------------------------------------------

    def _compute_fabric_indices(self, selection) -> wp.array:
        """View-side indices that map each managed prim into ``selection``."""
        return self._compute_fabric_indices_for(selection, list(self.prim_paths))

    def _compute_parent_fabric_indices(self, selection) -> wp.array:
        """View-side indices that map each managed prim's parent into ``selection``."""

        def parent_path(prim_path: str) -> str:
            p = prim_path.rsplit("/", 1)[0]
            if not p:
                raise RuntimeError(
                    f"Child prim '{prim_path}' is at stage root and has no parent prim. "
                    "FabricFrameView requires every prim to have a non-pseudoroot parent "
                    "with Fabric world+local matrices."
                )
            return p

        return self._compute_fabric_indices_for(selection, [parent_path(p) for p in self.prim_paths])

    def _build_indexed_array(self, selection, attribute_name: str, fabric_indices: wp.array) -> wp.indexedfabricarray:
        fa = wp.fabricarray(selection, attribute_name)
        return wp.indexedfabricarray(fa=fa, indices=fabric_indices)

    def _resolve_indices_wp(self, indices: wp.array | None) -> wp.array:
        """Resolve view indices as a Warp uint32 array."""
        if indices is None or indices == slice(None):
            if self._view_indices is None:
                raise RuntimeError("Fabric view indices are not initialized.")
            return self._view_indices
        if indices.dtype != wp.uint32:
            return wp.array(indices.numpy().astype("uint32"), dtype=wp.uint32, device=self._device)
        return indices

    # ------------------------------------------------------------------
    # Internal -- Fabric initialization
    # ------------------------------------------------------------------

    def _initialize_fabric(self) -> None:
        """One-time Fabric setup: hierarchy handle, attribute population, selections, indexed arrays."""
        import usdrt  # noqa: PLC0415
        from usdrt import Rt  # noqa: PLC0415

        from isaaclab.sim.utils import get_current_stage_id  # noqa: PLC0415

        # Attach usdrt stage and create hierarchy handle.
        stage_id = get_current_stage_id()
        self._stage = usdrt.Usd.Stage.Attach(stage_id)
        fabric_id = self._stage.GetFabricId()
        self._fabric_id = fabric_id.id
        self._fabric_hierarchy = usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(
            fabric_id, self._stage.GetStageIdAsStageId()
        )

        # Ensure each child prim AND its parent have BOTH Fabric world and local matrix
        # attributes.  ``Create*Attr`` calls are idempotent.
        seen_paths: set[str] = set()
        for child_path in self.prim_paths:
            for path in (child_path, child_path.rsplit("/", 1)[0]):
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                rt_prim = self._stage.GetPrimAtPath(path)
                if not rt_prim.IsValid():
                    continue
                rt_xformable = Rt.Xformable(rt_prim)
                rt_xformable.CreateFabricHierarchyWorldMatrixAttr()
                rt_xformable.CreateFabricHierarchyLocalMatrixAttr()
                rt_xformable.SetLocalXformFromUsd()
                rt_xformable.SetWorldXformFromUsd()

        # Two persistent selections: all-RO (steady state) and all-RW (active
        # only inside a writer scope).  Each will own its own bundle of
        # indexed-fabric arrays built lazily by ``_rebuild_{ro,rw}_arrays``.
        matrix = usdrt.Sdf.ValueTypeNames.Matrix4d
        ro = usdrt.Usd.Access.Read
        rw = usdrt.Usd.Access.ReadWrite
        wm_ro = (matrix, self._WORLD_MATRIX_NAME, ro)
        lm_ro = (matrix, self._LOCAL_MATRIX_NAME, ro)
        wm_rw = (matrix, self._WORLD_MATRIX_NAME, rw)
        lm_rw = (matrix, self._LOCAL_MATRIX_NAME, rw)
        self._sel_ro = self._stage.SelectPrims(require_attrs=[wm_ro, lm_ro], device=self._device, want_paths=True)
        self._sel_rw = self._stage.SelectPrims(require_attrs=[wm_rw, lm_rw], device=self._device, want_paths=True)

        self._view_indices = wp.array(list(range(self.count)), dtype=wp.uint32, device=self._device)
        self._rebuild_ro_arrays()
        self._rebuild_rw_arrays()

        # Pre-allocated reusable output buffers (world + local + scales).
        self._fabric_positions_buf = wp.zeros((self.count, 3), dtype=wp.float32, device=self._device)
        self._fabric_orientations_buf = wp.zeros((self.count, 4), dtype=wp.float32, device=self._device)
        self._fabric_scales_buf = wp.zeros((self.count, 3), dtype=wp.float32, device=self._device)
        self._fabric_local_translations_buf = wp.zeros((self.count, 3), dtype=wp.float32, device=self._device)
        self._fabric_local_orientations_buf = wp.zeros((self.count, 4), dtype=wp.float32, device=self._device)
        self._fabric_empty_2d_array_sentinel = wp.zeros((0, 0), dtype=wp.float32, device=self._device)

        self._fabric_positions_ta = ProxyArray(self._fabric_positions_buf)
        self._fabric_orientations_ta = ProxyArray(self._fabric_orientations_buf)
        self._fabric_scales_ta = ProxyArray(self._fabric_scales_buf)
        self._fabric_local_translations_ta = ProxyArray(self._fabric_local_translations_buf)
        self._fabric_local_orientations_ta = ProxyArray(self._fabric_local_orientations_buf)

        self._fabric_initialized = True

        # Seed Fabric matrices from USD authoritatively.  The seed writes, so
        # flip into the RW bundle for its duration; flip back to RO afterwards
        # so steady-state getters use the RO bundle.
        self._is_rw = True
        try:
            self._sync_fabric_from_usd_initial()
        finally:
            self._is_rw = False

    def _sync_fabric_from_usd_initial(self) -> None:
        """Populate Fabric world+local matrices for children and parents from USD.

        Performed once during ``_initialize_fabric``.  Without this step Fabric's
        matrices are identity for stages that haven't been rendered yet, and our
        getters (which read from Fabric) would return wrong values.
        """
        # --- Children: compose child localMatrix from USD-authored local transforms.
        scales_wp = _to_float32_2d(self._usd_view.get_local_scales().warp)
        local_pos_ta, local_ori_ta = self._usd_view.get_local_poses()
        wp.launch(
            kernel=fabric_utils.compose_indexed_fabric_transforms,
            dim=self.count,
            inputs=[
                self._local_ifa_rw,  # explicit RW: init-time write, no scope yet
                _to_float32_2d(local_pos_ta.warp),
                _to_float32_2d(local_ori_ta.warp),
                _to_float32_2d(scales_wp),
                False,
                False,
                False,
                self._view_indices,
            ],
            device=self._device,
        )

        # --- Parents (one entry per unique parent path) ---
        unique_parent_paths = list(dict.fromkeys(p.rsplit("/", 1)[0] for p in self.prim_paths))
        if unique_parent_paths:
            from isaaclab.sim.utils import get_current_stage  # noqa: PLC0415

            usd_stage = get_current_stage()
            xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            world_pos_rows: list[list[float]] = []
            world_ori_rows: list[list[float]] = []
            world_scale_rows: list[list[float]] = []
            decomposer = Gf.Transform()
            warned_shear = False
            for path in unique_parent_paths:
                prim = usd_stage.GetPrimAtPath(path)
                tf = xform_cache.GetLocalToWorldTransform(prim)
                decomposer.SetMatrix(tf)
                s = decomposer.GetScale()
                if not warned_shear:
                    row0 = Gf.Vec3d(tf[0][0], tf[0][1], tf[0][2]).GetNormalized()
                    row1 = Gf.Vec3d(tf[1][0], tf[1][1], tf[1][2]).GetNormalized()
                    row2 = Gf.Vec3d(tf[2][0], tf[2][1], tf[2][2]).GetNormalized()
                    if (
                        abs(Gf.Dot(row0, row1)) > 1e-3
                        or abs(Gf.Dot(row0, row2)) > 1e-3
                        or abs(Gf.Dot(row1, row2)) > 1e-3
                    ):
                        warned_shear = True
                        logger.warning(
                            "FabricFrameView: parent prim '%s' has a sheared/skewed world "
                            "transform. TRS decomposition (used by scale getters and world<->local "
                            "propagation) does not support shear -- extracted scales and rotations "
                            "will be approximate. Avoid shear in parent transforms for correct results.",
                            path,
                        )
                tf.Orthonormalize()
                t = tf.ExtractTranslation()
                q = tf.ExtractRotationQuat()
                img, real = q.GetImaginary(), q.GetReal()
                world_pos_rows.append([float(t[0]), float(t[1]), float(t[2])])
                world_ori_rows.append([float(img[0]), float(img[1]), float(img[2]), float(real)])
                world_scale_rows.append([float(s[0]), float(s[1]), float(s[2])])
            parent_view_indices = wp.array(list(range(len(unique_parent_paths))), dtype=wp.uint32, device=self._device)
            parent_pos_wp = wp.array(world_pos_rows, dtype=wp.float32, device=self._device)
            parent_ori_wp = wp.array(world_ori_rows, dtype=wp.float32, device=self._device)
            parent_scale_wp = wp.array(world_scale_rows, dtype=wp.float32, device=self._device)
            parent_world_rw = wp.indexedfabricarray(
                fa=wp.fabricarray(self._sel_rw, self._WORLD_MATRIX_NAME),
                indices=self._compute_fabric_indices_for(self._sel_rw, unique_parent_paths),
            )
            wp.launch(
                kernel=fabric_utils.compose_indexed_fabric_transforms,
                dim=len(unique_parent_paths),
                inputs=[
                    parent_world_rw,
                    parent_pos_wp,
                    parent_ori_wp,
                    parent_scale_wp,
                    False,
                    False,
                    False,
                    parent_view_indices,
                ],
                device=self._device,
            )
        wp.synchronize()

        # After seeding local matrices from USD, recompute world matrices so
        # the view starts with consistent state.
        self._recompute_world_from_local_all()
        wp.synchronize()

    def _compute_fabric_indices_for(self, selection, paths: list[str]) -> wp.array:
        """Look up each path in ``selection`` and return the matching fabric-side indices.

        Shared primitive used by :meth:`_compute_fabric_indices` (children),
        :meth:`_compute_parent_fabric_indices` (parents), and one-off
        index arrays such as the parent-world seed in
        :meth:`_sync_fabric_from_usd_initial`.
        """
        path_to_idx = {str(p): i for i, p in enumerate(selection.GetPaths())}

        def lookup(path: str) -> int:
            idx = path_to_idx.get(path)
            if idx is None:
                raise RuntimeError(f"Path '{path}' not found in Fabric selection.")
            return idx

        return wp.array([lookup(p) for p in paths], dtype=wp.int32, device=self._device)


# ----------------------------------------------------------------------
# Concrete writer classes for FabricFrameView
# ----------------------------------------------------------------------


class _FabricWriterMixin:
    """Common ``__enter__`` / ``__exit__`` for the Fabric world / local writers.

    On enter: pauses ``track_local_xform_changes`` / ``track_world_xform_changes``
    on the Fabric hierarchy (saving prior state) and flips the view's
    ``_is_rw`` so all get/set helpers resolve to the persistent RW selection
    bundle (no rebuild -- both bundles are kept alive for the view's lifetime).

    On exit (normal or via exception): runs a best-effort opposite-space
    derive + ``wp.synchronize()`` whenever any write happened inside the
    scope, then flips ``_is_rw`` back to ``False`` (RO bundle for
    steady-state reads) and restores hierarchy-tracking state.

    **Exception safety.** If the scope unwinds because of an exception
    (including ``KeyboardInterrupt`` from an interactive notebook), the
    opposite-space derive still runs so that ``worldMatrix`` and
    ``localMatrix`` are mutually consistent prim-by-prim on whatever
    partial-write state Fabric currently holds.  The partial write itself
    is **not** rolled back -- some prims may carry the new value and the
    rest the pre-scope value -- so callers needing transactional
    all-or-nothing semantics should snapshot matrices themselves before
    entering the scope.  If the recovery launch itself fails (typically
    because the original exception came from a poisoned CUDA stream), the
    failure is logged and the original exception propagates; the view
    should then be recreated.
    """

    def _enter_impl(self) -> None:
        view: FabricFrameView = self._view  # type: ignore[assignment]
        if not view._fabric_initialized:
            view._initialize_fabric()
        self._wrote_anything = False
        h = view._fabric_hierarchy
        self._was_tracking_local = h.tracking_local_xform_changes
        self._was_tracking_world = h.tracking_world_xform_changes
        if self._was_tracking_local:
            h.track_local_xform_changes(False)
        if self._was_tracking_world:
            h.track_world_xform_changes(False)
        view._is_rw = True

    def _exit_impl(self, exc_type, exc_val, exc_tb) -> None:
        view: FabricFrameView = self._view  # type: ignore[assignment]
        try:
            if self._wrote_anything:
                try:
                    self._derive_opposite()
                    wp.synchronize()
                except Exception as recovery_exc:
                    # Recovery itself failed (e.g. original exception came
                    # from a device error and the CUDA stream is poisoned).
                    # If we got here on the happy path, re-raise.  If we are
                    # already unwinding an exception, log and let the original
                    # propagate -- masking it would hide the actual root cause.
                    if exc_type is None:
                        raise
                    logger.error(
                        "FabricFrameView writer scope: best-effort opposite-space sync "
                        "failed during exception handling: %s. World/local matrices may "
                        "be inconsistent prim-by-prim; recreate the view to recover.",
                        recovery_exc,
                    )
        finally:
            # Flip back to RO before restoring hierarchy tracking so any
            # subsequent updateWorldXforms tick sees a fully-RO selection.
            view._is_rw = False
            h = view._fabric_hierarchy
            if self._was_tracking_world:
                h.track_world_xform_changes(True)
            if self._was_tracking_local:
                h.track_local_xform_changes(True)

    def _derive_opposite(self) -> None:
        raise NotImplementedError


class _FabricWorldSpaceWriter(_FabricWriterMixin, FrameViewWorldSpaceWriter):
    """World-space writer for :class:`FabricFrameView`.

    Writes flow through ``_world_ifa_rw`` (the RW-bundle worldMatrix array);
    on exit ``localMatrix`` is derived from the just-written ``worldMatrix``
    via :func:`update_indexed_local_matrix_from_world`.
    """

    def _derive_opposite(self) -> None:
        self._view._recompute_local_from_world_all()  # type: ignore[attr-defined]

    def set_poses(self, positions=None, orientations=None, indices=None) -> None:
        view: FabricFrameView = self._view  # type: ignore[assignment]
        indices_wp = view._resolve_indices_wp(indices)
        positions_wp = view._to_float32_2d_or_empty(positions)
        orientations_wp = view._to_float32_2d_or_empty(orientations)
        wp.launch(
            kernel=fabric_utils.compose_indexed_fabric_transforms,
            dim=indices_wp.shape[0],
            inputs=[
                view._get_world_ifa(),
                positions_wp,
                orientations_wp,
                view._fabric_empty_2d_array_sentinel,
                False,
                False,
                False,
                indices_wp,
            ],
            device=view._device,
        )
        self._wrote_anything = True

    def set_scales(self, scales, indices=None) -> None:
        view: FabricFrameView = self._view  # type: ignore[assignment]
        indices_wp = view._resolve_indices_wp(indices)
        scales_wp = view._to_float32_2d_or_empty(scales)
        wp.launch(
            kernel=fabric_utils.compose_indexed_fabric_transforms,
            dim=indices_wp.shape[0],
            inputs=[
                view._get_world_ifa(),
                view._fabric_empty_2d_array_sentinel,
                view._fabric_empty_2d_array_sentinel,
                scales_wp,
                False,
                False,
                False,
                indices_wp,
            ],
            device=view._device,
        )
        self._wrote_anything = True

    def get_poses(self, indices=None) -> tuple[ProxyArray, ProxyArray]:
        return self._view._get_world_poses_impl(indices)  # type: ignore[attr-defined]

    def get_scales(self, indices=None) -> ProxyArray:
        return self._view._get_world_scales_impl(indices)  # type: ignore[attr-defined]


class _FabricLocalSpaceWriter(_FabricWriterMixin, FrameViewLocalSpaceWriter):
    """Local-space writer for :class:`FabricFrameView`.

    Writes flow through ``_local_ifa_rw`` (the RW-bundle localMatrix array);
    on exit ``worldMatrix`` is derived from the just-written ``localMatrix``
    via :func:`update_indexed_world_matrix_from_local`.
    """

    def _derive_opposite(self) -> None:
        self._view._recompute_world_from_local_all()  # type: ignore[attr-defined]

    def set_poses(self, positions=None, orientations=None, indices=None) -> None:
        view: FabricFrameView = self._view  # type: ignore[assignment]
        indices_wp = view._resolve_indices_wp(indices)
        translations_wp = view._to_float32_2d_or_empty(positions)
        orientations_wp = view._to_float32_2d_or_empty(orientations)
        wp.launch(
            kernel=fabric_utils.compose_indexed_fabric_transforms,
            dim=indices_wp.shape[0],
            inputs=[
                view._get_local_ifa(),
                translations_wp,
                orientations_wp,
                view._fabric_empty_2d_array_sentinel,
                False,
                False,
                False,
                indices_wp,
            ],
            device=view._device,
        )
        self._wrote_anything = True

    def set_scales(self, scales, indices=None) -> None:
        view: FabricFrameView = self._view  # type: ignore[assignment]
        indices_wp = view._resolve_indices_wp(indices)
        scales_wp = view._to_float32_2d_or_empty(scales)
        wp.launch(
            kernel=fabric_utils.compose_indexed_fabric_transforms,
            dim=indices_wp.shape[0],
            inputs=[
                view._get_local_ifa(),
                view._fabric_empty_2d_array_sentinel,
                view._fabric_empty_2d_array_sentinel,
                scales_wp,
                False,
                False,
                False,
                indices_wp,
            ],
            device=view._device,
        )
        self._wrote_anything = True

    def get_poses(self, indices=None) -> tuple[ProxyArray, ProxyArray]:
        return self._view._get_local_poses_impl(indices)  # type: ignore[attr-defined]

    def get_scales(self, indices=None) -> ProxyArray:
        return self._view._get_local_scales_impl(indices)  # type: ignore[attr-defined]


class _FabricFallbackWorldWriter(FrameViewWorldSpaceWriter):
    """Fallback world-space writer used when Fabric is disabled.

    Delegates set/get calls to the internal :class:`UsdFrameView`'s backend
    hooks directly.  No batching, no listener pausing -- there's no Fabric to
    confuse.
    """

    def set_poses(self, positions=None, orientations=None, indices=None) -> None:
        self._view._usd_view._apply_world_pose_write(positions, orientations, indices)  # type: ignore[attr-defined]

    def set_scales(self, scales, indices=None) -> None:
        self._view._usd_view._apply_world_scale_write(scales, indices)  # type: ignore[attr-defined]

    def get_poses(self, indices=None) -> tuple[ProxyArray, ProxyArray]:
        return self._view._usd_view._get_world_poses_impl(indices)  # type: ignore[attr-defined]

    def get_scales(self, indices=None) -> ProxyArray:
        return self._view._usd_view._get_world_scales_impl(indices)  # type: ignore[attr-defined]


class _FabricFallbackLocalWriter(FrameViewLocalSpaceWriter):
    """Fallback local-space writer used when Fabric is disabled."""

    def set_poses(self, positions=None, orientations=None, indices=None) -> None:
        self._view._usd_view._apply_local_pose_write(positions, orientations, indices)  # type: ignore[attr-defined]

    def set_scales(self, scales, indices=None) -> None:
        self._view._usd_view._apply_local_scale_write(scales, indices)  # type: ignore[attr-defined]

    def get_poses(self, indices=None) -> tuple[ProxyArray, ProxyArray]:
        return self._view._usd_view._get_local_poses_impl(indices)  # type: ignore[attr-defined]

    def get_scales(self, indices=None) -> ProxyArray:
        return self._view._usd_view._get_local_scales_impl(indices)  # type: ignore[attr-defined]
