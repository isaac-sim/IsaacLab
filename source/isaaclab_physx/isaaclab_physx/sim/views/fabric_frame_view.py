# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX FrameView with Fabric GPU acceleration."""

from __future__ import annotations

import contextlib
import itertools
import logging
import sys

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


def _parent_path(prim_path: str) -> str:
    """Parent prim path of ``prim_path``.

    Args:
        prim_path: Absolute prim path, so it always contains a separator.

    Raises:
        RuntimeError: If the prim is directly under the stage root and thus has
            no non-pseudoroot parent to read Fabric matrices from.
    """
    parent = prim_path[: prim_path.rfind("/")]
    if not parent:
        raise RuntimeError(
            f"Child prim '{prim_path}' is at the stage root and has no parent prim. "
            "FabricFrameView requires every prim to have a non-pseudoroot parent "
            "with Fabric world+local matrices."
        )
    return parent


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
    * **Selections are scoped to the view, not the stage.**  The view tags its
      own prims (and their parents) with private per-view index attributes and
      requires those attributes in every prim selection, so a selection resolves
      to exactly the prims the view manages however large the stage grows.
      Tag names are unique per view instance, so views never interfere with one
      another.  The tags are authored on first use and removed again by
      :meth:`close` -- or, best-effort and with a warning, when the view is
      garbage collected.  Call :meth:`close` when done with a view;
      collection timing is up to the interpreter, so relying on it can remove
      the tags at an arbitrary point in the frame (or, on a leaked reference,
      not at all).
    * **Topology changes are absorbed, with no cache to invalidate.**  The
      view-to-Fabric mapping is re-derived from live Fabric data on every
      access, so prims moving between Fabric buckets can never leave a stale
      mapping behind.  If a managed prim disappears (prim or attribute removed)
      the next access raises :class:`RuntimeError` and the view must be
      recreated.  See ``_refresh_child_selection`` for how this is done.

    Pose getters return :class:`~isaaclab.utils.warp.ProxyArray`; the
    convenience :meth:`set_world_poses` / :meth:`set_local_poses` helpers accept
    :class:`wp.array`.  Inside a writer scope, the writer's
    :meth:`~FrameViewSpaceWriterBase.set_poses` / :meth:`~FrameViewSpaceWriterBase.set_scales`
    accept :class:`wp.array`.
    """

    _WORLD_MATRIX_NAME = "omni:fabric:worldMatrix"
    _LOCAL_MATRIX_NAME = "omni:fabric:localMatrix"

    # Process-wide uid source for per-view Fabric attribute names.  A monotonic
    # counter (NOT ``id(self)``/``hash(self)``) guarantees a name is never
    # reused after a view is garbage-collected, so a dead view's leftover
    # attributes can never satisfy a live view's selection.
    _view_uid_counter = itertools.count()

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

        # Per-view Fabric index attributes (authored once in ``_initialize_fabric``).
        self._child_index_attr: str | None = None
        self._parent_index_attr: str | None = None
        self._unique_parent_paths: list[str] = []

        # Three persistent selections keyed on the index attributes: child RO
        # (steady state), child RW (active inside a writer scope; ``_is_rw``
        # flips between them), and parent world (always read-only).
        self._sel_ro = None
        self._sel_rw = None
        self._sel_parent = None
        self._is_rw: bool = False

        # View-side indices array.
        self._view_indices: wp.array | None = None

        # Kernel-built view->fabric slot mappings, refreshed on every selection
        # access (see ``_refresh_child_selection``).  ``_child_parent_map`` holds
        # view-side indices (uint32, like the Fabric ``UInt`` index attributes);
        # the ``*_slots_buf`` buffers hold Fabric slots and must be int32, the
        # only dtype ``wp.indexedfabricarray`` accepts for indices.
        self._child_parent_map: wp.array | None = None
        self._child_slots_buf: wp.array | None = None
        self._parent_slots_buf: wp.array | None = None
        self._parent_slot_of_child_buf: wp.array | None = None

        # Sentinel passed to compose/decompose kernels for unused slots.
        self._fabric_empty_2d_array_sentinel: wp.array | None = None

        # Index-attribute cleanup state (see ``close``): the ``(attribute,
        # prims)`` groups authored by ``_initialize_fabric``, and the flag that
        # makes ``close()`` idempotent and lets ``__del__`` warn when cleanup
        # had to happen via garbage collection.
        self._tagged_prims: list[tuple[str, list]] = []
        self._is_closed: bool = False

    def close(self) -> None:
        """Remove this view's Fabric index attributes. The view must not be used afterwards.

        Calling :meth:`close` again is a no-op.  If :meth:`close` is never
        called, the same cleanup runs best-effort from ``__del__`` (with a
        warning, since collection timing is up to the interpreter) -- except at
        interpreter exit, where Fabric is being torn down anyway and the
        attributes die with it.
        """
        if self._is_closed:
            return
        self._is_closed = True
        failed = total = 0
        for attr, prims in self._tagged_prims:
            total += len(prims)
            for prim in prims:
                try:
                    prim.RemoveProperty(attr)
                except Exception:  # noqa: BLE001 -- one bad handle must not strand the remaining tags
                    failed += 1
        self._tagged_prims = []
        if failed:
            logger.debug("FabricFrameView(%s): %d of %d tag removals failed", self._usd_view._prim_path, failed, total)

    def __del__(self, _sys=sys):
        """Best-effort cleanup when the view is collected without :meth:`close`.

        Follows the repo's shutdown-safe ``__del__`` idiom (see
        :meth:`~isaaclab.envs.ManagerBasedEnv.__del__`): ``sys`` is bound as a
        default argument so it survives module teardown, and nothing runs during
        interpreter finalization, when calling into Kit can crash and the
        attributes die with Fabric anyway.
        """
        # getattr: __init__ may have raised before the flag existed
        if getattr(self, "_is_closed", True) or _sys.is_finalizing() or _sys.meta_path is None:
            return
        if self._tagged_prims:
            logger.warning(
                "FabricFrameView(%s) was garbage-collected without close(); its Fabric index "
                "attributes were removed best-effort at an arbitrary point in the frame. Call "
                "close() for deterministic cleanup.",
                self._usd_view._prim_path,
            )
        with contextlib.suppress(Exception):  # never propagate from __del__
            self.close()

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
        world_ifa, local_ifa = self._get_child_ifas()
        wp.launch(
            kernel=fabric_utils.update_indexed_local_matrix_from_world,
            dim=self.count,
            inputs=[
                world_ifa,
                self._get_parent_world_ifa(),
                local_ifa,
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
        world_ifa, local_ifa = self._get_child_ifas()
        wp.launch(
            kernel=fabric_utils.update_indexed_world_matrix_from_local,
            dim=self.count,
            inputs=[
                local_ifa,
                self._get_parent_world_ifa(),
                world_ifa,
                self._view_indices,
            ],
            device=self._device,
        )

    # ------------------------------------------------------------------
    # Internal -- selection accessors (kernel-built slot mappings)
    # ------------------------------------------------------------------

    def _get_world_ifa(self) -> wp.indexedfabricarray:
        sel = self._refresh_child_selection()
        return wp.indexedfabricarray(fa=wp.fabricarray(sel, self._WORLD_MATRIX_NAME), indices=self._child_slots_buf)

    def _get_local_ifa(self) -> wp.indexedfabricarray:
        sel = self._refresh_child_selection()
        return wp.indexedfabricarray(fa=wp.fabricarray(sel, self._LOCAL_MATRIX_NAME), indices=self._child_slots_buf)

    def _get_child_ifas(self) -> tuple[wp.indexedfabricarray, wp.indexedfabricarray]:
        """Return ``(world, local)`` child arrays from a single selection refresh.

        Callers that need both spaces must use this instead of calling
        :meth:`_get_world_ifa` and :meth:`_get_local_ifa`, which would refresh
        the same selection -- and re-run its mapping kernel -- twice.
        """
        sel = self._refresh_child_selection()
        return (
            wp.indexedfabricarray(fa=wp.fabricarray(sel, self._WORLD_MATRIX_NAME), indices=self._child_slots_buf),
            wp.indexedfabricarray(fa=wp.fabricarray(sel, self._LOCAL_MATRIX_NAME), indices=self._child_slots_buf),
        )

    def _get_parent_world_ifa(self) -> wp.indexedfabricarray:
        self._refresh_parent_selection()
        return wp.indexedfabricarray(
            fa=wp.fabricarray(self._sel_parent, self._WORLD_MATRIX_NAME),
            indices=self._parent_slot_of_child_buf,
        )

    def _refresh_child_selection(self):
        """Refresh the active child selection and rebuild its slot mapping on device.

        Runs on every accessor call.  ``PrepareForReuse`` lets the persistent
        selection absorb Fabric bucket changes (and notifies the renderer for
        the RW selection); a single Warp kernel launch over the selection's
        index attribute then rebuilds ``_child_slots_buf`` so that entry ``i``
        is the fabric-side slot of view prim ``i``.  Re-deriving the mapping
        from live Fabric data on each access means bucket reorders can never
        leave a stale mapping behind, with no host-side path resolution and no
        cache to invalidate.

        Returns:
            The active (RO or RW) child prim selection.
        """
        sel = self._sel_rw if self._is_rw else self._sel_ro
        sel.PrepareForReuse()
        self._check_selection_count(sel.GetCount(), self.count, self._child_index_attr)
        wp.launch(
            kernel=fabric_utils.map_view_indices_to_fabric_slots,
            dim=self.count,
            inputs=[wp.fabricarray(sel, self._child_index_attr), self._child_slots_buf],
            device=self._device,
        )
        return sel

    def _refresh_parent_selection(self) -> None:
        """Refresh the parent selection and rebuild the per-child parent-slot mapping.

        Two kernel launches: the first inverts the parent index attribute into
        per-ordinal fabric slots, the second gathers those slots per child
        through ``_child_parent_map`` (children sharing a parent read the same
        slot).
        """
        num_parents = self._parent_slots_buf.shape[0]
        self._sel_parent.PrepareForReuse()
        self._check_selection_count(self._sel_parent.GetCount(), num_parents, self._parent_index_attr)
        wp.launch(
            kernel=fabric_utils.map_view_indices_to_fabric_slots,
            dim=num_parents,
            inputs=[wp.fabricarray(self._sel_parent, self._parent_index_attr), self._parent_slots_buf],
            device=self._device,
        )
        wp.launch(
            kernel=fabric_utils.gather_fabric_slots,
            dim=self.count,
            inputs=[self._parent_slots_buf, self._child_parent_map, self._parent_slot_of_child_buf],
            device=self._device,
        )

    def _check_selection_count(self, found: int, expected: int, index_attr: str) -> None:
        """Raise if a selection stopped matching exactly the view's tagged prims."""
        if found != expected:
            raise RuntimeError(
                f"FabricFrameView: selection on '{index_attr}' matched {found} prims, expected {expected}. "
                "A prim managed by this view (or one of its Fabric matrix/index attributes) was removed "
                "from the Fabric stage; recreate the view."
            )

    def _resolve_indices_wp(self, indices: wp.array | None) -> wp.array:
        """Resolve view indices as a Warp uint32 array."""
        if indices is None or indices == slice(None):
            if self._view_indices is None:
                raise RuntimeError("Fabric view indices are not initialized.")
            return self._view_indices
        if indices.dtype == wp.uint32:
            return indices
        if indices.dtype == wp.int32:
            # Zero-copy reinterpret: callers (e.g. Camera) pass non-negative int32 indices.
            # Device placement is not checked here; ``wp.launch`` validates it for every input.
            return indices.view(wp.uint32)
        return wp.array(indices.numpy().astype("uint32"), dtype=wp.uint32, device=self._device)

    # ------------------------------------------------------------------
    # Internal -- Fabric initialization
    # ------------------------------------------------------------------
    def _initialize_fabric(self) -> None:
        """One-time Fabric setup: hierarchy handle, per-view index tagging, selections, buffers."""
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

        # Per-view Fabric index attribute names (see ``_view_uid_counter``).
        uid = next(FabricFrameView._view_uid_counter)
        self._child_index_attr = f"isaaclab:fabricFrameView:{uid}:index"
        self._parent_index_attr = f"isaaclab:fabricFrameView:{uid}:parentIndex"

        # Per-child parent paths, computed once and reused for the ordinal map
        # below.  Unique parents keep first-occurrence order; ``parent_ordinal``
        # maps a parent path to its position in that order.
        child_parent_paths = [_parent_path(p) for p in self.prim_paths]
        self._unique_parent_paths = list(dict.fromkeys(child_parent_paths))
        parent_ordinal = {path: i for i, path in enumerate(self._unique_parent_paths)}

        # Tag children and parents with their per-view index and ensure both
        # carry the Fabric world+local matrix attributes (``Create*Attr`` calls
        # are idempotent).  The index attribute doubles as the selection filter:
        # the selections below match ONLY tagged prims, so their size is
        # O(view), not O(stage).  A prim that is both a child and a parent of
        # this view receives both index attributes.
        tagged_prims: list[tuple[str, list]] = []
        for paths, index_attr in (
            (list(self.prim_paths), self._child_index_attr),
            (self._unique_parent_paths, self._parent_index_attr),
        ):
            group_prims: list = []
            for i, path in enumerate(paths):
                rt_prim = self._stage.GetPrimAtPath(path)
                if not rt_prim.IsValid():
                    raise RuntimeError(f"FabricFrameView: prim '{path}' does not exist in the Fabric stage.")
                rt_xformable = Rt.Xformable(rt_prim)
                rt_xformable.CreateFabricHierarchyWorldMatrixAttr()
                rt_xformable.CreateFabricHierarchyLocalMatrixAttr()
                rt_xformable.SetLocalXformFromUsd()
                rt_xformable.SetWorldXformFromUsd()
                rt_prim.CreateAttribute(index_attr, usdrt.Sdf.ValueTypeNames.UInt, custom=True)
                rt_prim.GetAttribute(index_attr).Set(i)
                group_prims.append(rt_prim)
            tagged_prims.append((index_attr, group_prims))

        # Remembered so ``close()`` / ``__del__`` can remove the tags again.
        self._tagged_prims = tagged_prims

        # Three persistent selections keyed on the per-view index attributes:
        # child RO (steady state), child RW (active only inside a writer
        # scope), and parent world (always read-only).
        matrix = usdrt.Sdf.ValueTypeNames.Matrix4d
        uint_type = usdrt.Sdf.ValueTypeNames.UInt
        ro = usdrt.Usd.Access.Read
        rw = usdrt.Usd.Access.ReadWrite
        child_tag = (uint_type, self._child_index_attr, ro)
        parent_tag = (uint_type, self._parent_index_attr, ro)
        wm_ro = (matrix, self._WORLD_MATRIX_NAME, ro)
        lm_ro = (matrix, self._LOCAL_MATRIX_NAME, ro)
        wm_rw = (matrix, self._WORLD_MATRIX_NAME, rw)
        lm_rw = (matrix, self._LOCAL_MATRIX_NAME, rw)
        self._sel_ro = self._stage.SelectPrims(require_attrs=[child_tag, wm_ro, lm_ro], device=self._device)
        self._sel_rw = self._stage.SelectPrims(require_attrs=[child_tag, wm_rw, lm_rw], device=self._device)
        self._sel_parent = self._stage.SelectPrims(require_attrs=[parent_tag, wm_ro], device=self._device)

        # View-side indices + kernel-built slot-mapping buffers.
        self._view_indices = wp.array(list(range(self.count)), dtype=wp.uint32, device=self._device)
        self._child_parent_map = wp.array(
            [parent_ordinal[p] for p in child_parent_paths], dtype=wp.uint32, device=self._device
        )
        self._child_slots_buf = wp.empty((self.count,), dtype=wp.int32, device=self._device)
        self._parent_slots_buf = wp.empty((len(self._unique_parent_paths),), dtype=wp.int32, device=self._device)
        self._parent_slot_of_child_buf = wp.empty((self.count,), dtype=wp.int32, device=self._device)

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
        # flip onto the RW selection for its duration; flip back afterwards so
        # steady-state getters use the RO selection.
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
                self._get_local_ifa(),  # caller holds ``_is_rw=True``: init-time write, no scope yet
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
        unique_parent_paths = self._unique_parent_paths
        if unique_parent_paths:
            import usdrt  # noqa: PLC0415

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
            # One-off RW selection on the parent tag for the initial seed; the
            # persistent ``_sel_parent`` stays read-only for steady-state reads.
            sel_parent_rw = self._stage.SelectPrims(
                require_attrs=[
                    (usdrt.Sdf.ValueTypeNames.UInt, self._parent_index_attr, usdrt.Usd.Access.Read),
                    (usdrt.Sdf.ValueTypeNames.Matrix4d, self._WORLD_MATRIX_NAME, usdrt.Usd.Access.ReadWrite),
                ],
                device=self._device,
            )
            self._check_selection_count(sel_parent_rw.GetCount(), len(unique_parent_paths), self._parent_index_attr)
            wp.launch(
                kernel=fabric_utils.map_view_indices_to_fabric_slots,
                dim=len(unique_parent_paths),
                inputs=[wp.fabricarray(sel_parent_rw, self._parent_index_attr), self._parent_slots_buf],
                device=self._device,
            )
            parent_world_rw = wp.indexedfabricarray(
                fa=wp.fabricarray(sel_parent_rw, self._WORLD_MATRIX_NAME),
                indices=self._parent_slots_buf,
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


# ----------------------------------------------------------------------
# Concrete writer classes for FabricFrameView
# ----------------------------------------------------------------------


class _FabricWriterMixin:
    """Common ``__enter__`` / ``__exit__`` for the Fabric world / local writers.

    On enter: pauses ``track_local_xform_changes`` / ``track_world_xform_changes``
    on the Fabric hierarchy (saving prior state) and flips the view's
    ``_is_rw`` so all get/set helpers resolve to the persistent RW selection
    (both selections are kept alive for the view's lifetime).

    On exit (normal or via exception): runs a best-effort opposite-space
    derive + ``wp.synchronize()`` whenever any write happened inside the
    scope, then flips ``_is_rw`` back to ``False`` (RO selection for
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

    Writes flow through the RW selection's ``worldMatrix`` indexed array;
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

    Writes flow through the RW selection's ``localMatrix`` indexed array;
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
