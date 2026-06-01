# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX FrameView with Fabric GPU acceleration."""

from __future__ import annotations

import enum
import logging

import torch
import warp as wp

from pxr import Gf, Usd, UsdGeom

from isaaclab.app.settings_manager import SettingsManager
from isaaclab.sim.views.base_frame_view import BaseFrameView
from isaaclab.sim.views.usd_frame_view import UsdFrameView
from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.warp import fabric as fabric_utils

logger = logging.getLogger(__name__)


class _DirtyFlag(enum.Enum):
    """Which matrix direction is stale and needs recomputation on the next read."""

    NONE = 0
    #: World matrices are stale (a prior ``set_local_poses`` wrote new locals).
    WORLD = 1
    #: Local matrices are stale (a prior ``set_world_poses``/``set_world_scales`` wrote new worlds).
    LOCAL = 2


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
    * **World ↔ local consistency (lazy).**  Getters are lazy: after
      ``set_world_poses`` or ``set_world_scales``, local matrices are only
      recomputed when ``get_local_poses`` (or ``get_local_scales``) is called;
      after ``set_local_poses`` or ``set_local_scales``, world matrices are
      only recomputed when ``get_world_poses`` (or ``get_world_scales``) is
      called.  Both directions stay in sync without round-tripping through USD.
    * **Dirty-flag invariant.**  The ``_dirty`` enum is one of ``NONE``,
      ``WORLD``, or ``LOCAL`` -- mutually exclusive by construction.
      ``set_world_poses`` / ``set_world_scales`` sets ``_dirty = LOCAL``;
      ``set_local_poses`` / ``set_local_scales`` sets ``_dirty = WORLD``.
      If the user interleaves both setters on the same view within a single
      frame, the second setter flushes the first's stale data before writing.
      This is correct but incurs an extra kernel launch -- a one-time warning
      is logged when this happens.
    * **Topology-adaptive.**  Fabric topology changes are detected on each
      access; the view rebuilds its internal mapping automatically and no
      manual refresh is required.  Steady-state overhead is negligible.

    Performance note:
        The fast path assumes the user calls **either** ``set_world_poses``
        **or** ``set_local_poses`` exclusively within a frame (not both).
        In that case, setters are O(1) kernel launches with no synchronization
        overhead beyond the single ``wp.synchronize()``; getters lazily flush
        the opposite direction only when actually needed.

        Interleaving both setters on different index subsets within the same
        frame is supported and correct, but triggers an extra flush kernel
        per transition.  A warning is emitted once per view instance.

    Pose getters return :class:`~isaaclab.utils.warp.ProxyArray`; setters
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
                device string (``"cuda:0"``, ``"cuda:1"``, …); Fabric
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
        # TODO(pv): Fuse set_world_poses/set_world_scales into single kernel launch (PR #5674 pv/fabric-fused-compose)

        self._fabric_initialized = False
        self._stage = None
        self._fabric_hierarchy = None
        # Tracks which matrix direction is stale.  Mutually exclusive by construction.
        # Per-view (not per-stage) so concurrent views on the same stage don't interfere.
        self._dirty: _DirtyFlag = _DirtyFlag.NONE
        self._warned_interleaved_set: bool = False

        # Selection (single RW covering both world + local matrix).
        self._sel = None

        # Index arrays (view-side indices and view->fabric mapping).
        self._view_indices: wp.array | None = None
        self._fabric_indices: wp.array | None = None

        # Indexed fabric arrays.
        self._world_ifa = None
        self._local_ifa = None
        self._parent_world_ifa = None

        # Sentinel passed to compose/decompose kernels for unused slots.
        # Kernels gate per-row access on ``shape[0] > 0``, so (0, 0) suffices.
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
    # World poses — Fabric-accelerated or USD fallback
    # ------------------------------------------------------------------

    def set_world_poses(self, positions=None, orientations=None, indices=None):
        if not self._use_fabric:
            self._usd_view.set_world_poses(positions, orientations, indices)
            return

        if not self._fabric_initialized:
            self._initialize_fabric()

        # If a prior set_local_poses left worlds stale, flush them now.
        if self._dirty == _DirtyFlag.WORLD and not self._warned_interleaved_set:
            self._warned_interleaved_set = True
            logger.warning(
                "FabricFrameView: set_world_poses called while world matrices are stale from a "
                "prior set_local_poses. Flushing stale worlds first. "
                "For best performance, avoid interleaving set_world_poses and set_local_poses "
                "on the same view within a single frame -- use one or the other exclusively."
            )

        self._sync_world_from_local_if_dirty()

        indices_wp = self._resolve_indices_wp(indices)
        positions_wp = self._to_float32_2d_or_empty(positions)
        orientations_wp = self._to_float32_2d_or_empty(orientations)

        wp.launch(
            kernel=fabric_utils.compose_indexed_fabric_transforms,
            dim=indices_wp.shape[0],
            inputs=[
                self._get_world_array(),
                positions_wp,
                orientations_wp,
                self._fabric_empty_2d_array_sentinel,
                False,
                False,
                False,
                indices_wp,
            ],
            device=self._device,
        )
        wp.synchronize()

        # World was just written -- mark local poses as stale so the next
        # get_local_poses recomputes them lazily.
        self._dirty = _DirtyFlag.LOCAL

    def get_world_poses(self, indices: wp.array | None = None) -> tuple[ProxyArray, ProxyArray]:
        """Return (positions, orientations) in world frame.

        .. warning::
            When *indices* is None (all prims), the returned arrays are **shared
            pre-allocated buffers** that are overwritten on the next call.  Do not
            hold references across calls -- copy if persistence is needed.
        """
        if not self._use_fabric:
            return self._usd_view.get_world_poses(indices)

        if not self._fabric_initialized:
            self._initialize_fabric()

        # If a prior set_local_poses left worldMatrix stale, propagate local -> world first.
        self._sync_world_from_local_if_dirty()

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
                self._get_world_array(),
                positions_wp,
                orientations_wp,
                self._fabric_empty_2d_array_sentinel,
                indices_wp,
            ],
            device=self._device,
        )

        if use_cached:
            wp.synchronize()
            return self._fabric_positions_ta, self._fabric_orientations_ta
        return ProxyArray(positions_wp), ProxyArray(orientations_wp)

    # ------------------------------------------------------------------
    # Local poses
    # ------------------------------------------------------------------

    def set_local_poses(self, translations=None, orientations=None, indices=None):
        if not self._use_fabric:
            self._usd_view.set_local_poses(translations, orientations, indices)
            return

        if not self._fabric_initialized:
            self._initialize_fabric()

        # If a prior set_world_poses left locals stale, flush them now before we
        # overwrite a (possibly different) subset of local matrices.
        if self._dirty == _DirtyFlag.LOCAL and not self._warned_interleaved_set:
            self._warned_interleaved_set = True
            logger.warning(
                "FabricFrameView: set_local_poses called while local matrices are stale from a "
                "prior set_world_poses/set_world_scales. Flushing stale locals first. "
                "For best performance, avoid interleaving set_world_poses and set_local_poses "
                "on the same view within a single frame -- use one or the other exclusively."
            )

        self._sync_local_from_world_if_dirty()

        indices_wp = self._resolve_indices_wp(indices)
        translations_wp = self._to_float32_2d_or_empty(translations)
        orientations_wp = self._to_float32_2d_or_empty(orientations)

        wp.launch(
            kernel=fabric_utils.compose_indexed_fabric_transforms,
            dim=indices_wp.shape[0],
            inputs=[
                self._get_local_array(),
                translations_wp,
                orientations_wp,
                self._fabric_empty_2d_array_sentinel,
                False,
                False,
                False,
                indices_wp,
            ],
            device=self._device,
        )
        wp.synchronize()

        # Mark this view's worlds stale so the next world read recomputes them.
        self._dirty = _DirtyFlag.WORLD

    def get_local_poses(self, indices: wp.array | None = None) -> tuple[ProxyArray, ProxyArray]:
        """Return (translations, orientations) in parent-local frame.

        .. warning::
            When *indices* is None (all prims), the returned arrays are **shared
            pre-allocated buffers** that are overwritten on the next call.  Do not
            hold references across calls -- copy if persistence is needed.
        """
        if not self._use_fabric:
            return self._usd_view.get_local_poses(indices)

        if not self._fabric_initialized:
            self._initialize_fabric()

        # If a prior set_world_poses/set_world_scales left localMatrix stale, recompute.
        self._sync_local_from_world_if_dirty()

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
                self._get_local_array(),
                translations_wp,
                orientations_wp,
                self._fabric_empty_2d_array_sentinel,
                indices_wp,
            ],
            device=self._device,
        )

        if use_cached:
            wp.synchronize()
            return self._fabric_local_translations_ta, self._fabric_local_orientations_ta
        return ProxyArray(translations_wp), ProxyArray(orientations_wp)

    # ------------------------------------------------------------------
    # Scales
    # ------------------------------------------------------------------

    def set_world_scales(self, scales, indices=None):
        """Set world-space (composed) scales by decomposing/recomposing worldMatrix."""
        if not self._use_fabric:
            self._usd_view.set_world_scales(scales, indices)
            return

        if not self._fabric_initialized:
            self._initialize_fabric()

        # Sync world matrices first if local writes are pending.
        self._sync_world_from_local_if_dirty()

        indices_wp = self._resolve_indices_wp(indices)
        scales_wp = self._to_float32_2d_or_empty(scales)

        wp.launch(
            kernel=fabric_utils.compose_indexed_fabric_transforms,
            dim=indices_wp.shape[0],
            inputs=[
                self._get_world_array(),
                self._fabric_empty_2d_array_sentinel,
                self._fabric_empty_2d_array_sentinel,
                scales_wp,
                False,
                False,
                False,
                indices_wp,
            ],
            device=self._device,
        )
        wp.synchronize()

        # World was just written -- mark local poses as stale.
        self._dirty = _DirtyFlag.LOCAL

    def get_world_scales(self, indices=None):
        """Return per-prim (sx, sy, sz) scales extracted from world matrix.

        .. warning::
            When *indices* is None (all prims), the returned array is a **shared
            pre-allocated buffer** (shared with :meth:`get_local_scales`) that is
            overwritten on the next call.  Do not hold references across calls --
            copy if persistence is needed.
        """
        if not self._use_fabric:
            return self._usd_view.get_world_scales(indices)

        if not self._fabric_initialized:
            self._initialize_fabric()

        # Sync world matrices first if local writes are pending.
        self._sync_world_from_local_if_dirty()

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
                self._get_world_array(),
                self._fabric_empty_2d_array_sentinel,
                self._fabric_empty_2d_array_sentinel,
                scales_wp,
                indices_wp,
            ],
            device=self._device,
        )

        if use_cached:
            wp.synchronize()
            return self._fabric_scales_ta
        return ProxyArray(scales_wp)

    def set_local_scales(self, scales, indices=None):
        """Set local-space scales by decomposing/recomposing localMatrix."""
        if not self._use_fabric:
            self._usd_view.set_local_scales(scales, indices)
            return

        if not self._fabric_initialized:
            self._initialize_fabric()

        # Sync local matrices first if world writes are pending.
        self._sync_local_from_world_if_dirty()

        indices_wp = self._resolve_indices_wp(indices)
        scales_wp = self._to_float32_2d_or_empty(scales)

        wp.launch(
            kernel=fabric_utils.compose_indexed_fabric_transforms,
            dim=indices_wp.shape[0],
            inputs=[
                self._get_local_array(),
                self._fabric_empty_2d_array_sentinel,
                self._fabric_empty_2d_array_sentinel,
                scales_wp,
                False,
                False,
                False,
                indices_wp,
            ],
            device=self._device,
        )
        wp.synchronize()

        # Local was just written -- mark world poses as stale.
        self._dirty = _DirtyFlag.WORLD

    def get_local_scales(self, indices=None):
        """Return per-prim (sx, sy, sz) scales extracted from local matrix.

        .. warning::
            When *indices* is None (all prims), the returned array is a **shared
            pre-allocated buffer** (shared with :meth:`get_world_scales`) that is
            overwritten on the next call.  Do not hold references across calls --
            copy if persistence is needed.
        """
        if not self._use_fabric:
            return self._usd_view.get_local_scales(indices)

        if not self._fabric_initialized:
            self._initialize_fabric()

        # Sync local matrices first if world writes are pending.
        self._sync_local_from_world_if_dirty()

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
                self._get_local_array(),
                self._fabric_empty_2d_array_sentinel,
                self._fabric_empty_2d_array_sentinel,
                scales_wp,
                indices_wp,
            ],
            device=self._device,
        )

        if use_cached:
            wp.synchronize()
            return self._fabric_scales_ta
        return ProxyArray(scales_wp)

    def _get_scales_impl(self, indices=None):
        """Fabric: deprecated get_scales delegates to get_world_scales."""
        return self.get_world_scales(indices).warp

    def _set_scales_impl(self, scales, indices=None):
        """Fabric: deprecated set_scales delegates to set_world_scales."""
        self.set_world_scales(scales, indices)

    # ------------------------------------------------------------------
    # Internal -- sync helpers
    # ------------------------------------------------------------------

    def _to_float32_2d_or_empty(self, data):
        return self._fabric_empty_2d_array_sentinel if data is None else _to_float32_2d(data)

    def _sync_world_from_local_if_dirty(self) -> None:
        """If a prior local write left world matrices stale, recompute them."""
        if self._dirty != _DirtyFlag.WORLD:
            return
        self._recompute_world_from_local()
        self._dirty = _DirtyFlag.NONE

    def _recompute_world_from_local(self) -> None:
        """Recompute world matrices: child_world = parent_world * child_local.

        We deliberately do NOT call ``IFabricHierarchy.update_world_xforms()`` --
        in practice that re-reads USD's authored xformOps and overwrites the Fabric
        local+world matrices we just authored.  Instead we fire a Warp kernel that
        does the multiply per child, leaving the Fabric-side localMatrix untouched.
        """
        self._refresh_if_needed()
        wp.launch(
            kernel=fabric_utils.update_indexed_world_matrix_from_local,
            dim=self.count,
            inputs=[
                self._local_ifa,
                self._parent_world_ifa,
                self._world_ifa,
                self._view_indices,
            ],
            device=self._device,
        )
        wp.synchronize()

    def _sync_local_from_world(self, indices_wp: wp.array) -> None:
        """Recompute child localMatrix from (parent worldMatrix, child worldMatrix).

        Called after ``set_world_poses`` so that subsequent ``get_local_poses`` returns
        values consistent with the just-written world poses.
        """
        self._refresh_if_needed()
        wp.launch(
            kernel=fabric_utils.update_indexed_local_matrix_from_world,
            dim=indices_wp.shape[0],
            inputs=[
                self._world_ifa,
                self._parent_world_ifa,
                self._local_ifa,
                indices_wp,
            ],
            device=self._device,
        )
        wp.synchronize()

    def _sync_local_from_world_if_dirty(self) -> None:
        """If a prior world write left local matrices stale, recompute them lazily."""
        if self._dirty != _DirtyFlag.LOCAL:
            return
        self._sync_local_from_world(self._view_indices)
        self._dirty = _DirtyFlag.NONE

    # ------------------------------------------------------------------
    # Internal -- selection accessors with on-demand index rebuild
    # ------------------------------------------------------------------

    def _refresh_if_needed(self):
        """Rebuild indexed arrays if the selection's prim set changed."""
        if self._sel.PrepareForReuse() or self._world_ifa is None:
            self._fabric_indices = self._compute_fabric_indices(self._sel)
            self._world_ifa = self._build_indexed_array(self._sel, self._WORLD_MATRIX_NAME, self._fabric_indices)
            self._local_ifa = self._build_indexed_array(self._sel, self._LOCAL_MATRIX_NAME, self._fabric_indices)
            self._parent_world_ifa = self._build_parent_indexed_array(self._sel)

    def _get_world_array(self):
        self._refresh_if_needed()
        return self._world_ifa

    def _get_local_array(self):
        self._refresh_if_needed()
        return self._local_ifa

    def _get_parent_world_array(self):
        self._refresh_if_needed()
        return self._parent_world_ifa

    # ------------------------------------------------------------------
    # Internal -- index computation
    # ------------------------------------------------------------------

    def _compute_fabric_indices(self, selection) -> wp.array:
        fabric_paths = selection.GetPaths()
        path_to_fabric_idx: dict[str, int] = {str(p): i for i, p in enumerate(fabric_paths)}
        indices: list[int] = []
        for prim_path in self.prim_paths:
            fabric_idx = path_to_fabric_idx.get(prim_path)
            if fabric_idx is None:
                raise RuntimeError(
                    f"Prim '{prim_path}' not found in Fabric selection. Ensure the hierarchy has been populated."
                )
            indices.append(fabric_idx)
        return wp.array(indices, dtype=wp.int32, device=self._device)

    def _compute_parent_fabric_indices(self, selection) -> wp.array:
        """For each child in this view, look up the parent prim's fabric index."""
        fabric_paths = selection.GetPaths()
        path_to_fabric_idx: dict[str, int] = {str(p): i for i, p in enumerate(fabric_paths)}
        indices: list[int] = []
        for prim_path in self.prim_paths:
            parent_path = prim_path.rsplit("/", 1)[0]
            if parent_path == "":
                raise RuntimeError(
                    f"Child prim '{prim_path}' is at stage root and has no parent prim. "
                    "FabricFrameView requires every prim to have a non-pseudoroot parent "
                    "with Fabric world+local matrices."
                )
            fabric_idx = path_to_fabric_idx.get(parent_path)
            if fabric_idx is None:
                raise RuntimeError(
                    f"Parent prim '{parent_path}' (for child '{prim_path}') not found in Fabric selection. "
                    "Ensure parents have Fabric world+local matrices populated."
                )
            indices.append(fabric_idx)
        return wp.array(indices, dtype=wp.int32, device=self._device)

    def _build_indexed_array(self, selection, attribute_name: str, fabric_indices: wp.array) -> wp.indexedfabricarray:
        fa = wp.fabricarray(selection, attribute_name)
        return wp.indexedfabricarray(fa=fa, indices=fabric_indices)

    def _build_parent_indexed_array(self, selection) -> wp.indexedfabricarray:
        self._parent_fabric_indices = self._compute_parent_fabric_indices(selection)
        fa = wp.fabricarray(selection, self._WORLD_MATRIX_NAME)
        return wp.indexedfabricarray(fa=fa, indices=self._parent_fabric_indices)

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
        # attributes.  Our ``trans_ro`` selection requires both, so prims missing either
        # would silently be excluded.  ``Create*Attr`` calls are idempotent.
        #
        # ``SetWorldXformFromUsd`` writes Fabric's worldMatrix from USD's accumulated
        # local-to-world transform (so it picks up the parent chain).
        # ``SetLocalXformFromUsd`` writes Fabric's localMatrix from USD's authored
        # xformOps on this prim only.  Calling both gives Fabric a consistent
        # (worldMatrix, localMatrix) pair for each prim before we touch the hierarchy.
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

        # Single RW selection covering both matrices.
        # TODO: Benchmark RO vs RW selection split -- separate RO selections could reduce
        # lock contention under concurrent Fabric access, but current usage is single-threaded.
        matrix = usdrt.Sdf.ValueTypeNames.Matrix4d
        rw = usdrt.Usd.Access.ReadWrite
        wm_rw = (matrix, self._WORLD_MATRIX_NAME, rw)
        lm_rw = (matrix, self._LOCAL_MATRIX_NAME, rw)
        self._sel = self._stage.SelectPrims(require_attrs=[wm_rw, lm_rw], device=self._device, want_paths=True)

        # Build the view-side indices array (just [0..count-1]) and a
        # view->fabric mapping (selections do not guarantee a shared path ordering).
        self._view_indices = wp.array(list(range(self.count)), dtype=wp.uint32, device=self._device)
        self._fabric_indices = self._compute_fabric_indices(self._sel)

        # Indexed fabric arrays per attribute.
        self._world_ifa = self._build_indexed_array(self._sel, self._WORLD_MATRIX_NAME, self._fabric_indices)
        self._local_ifa = self._build_indexed_array(self._sel, self._LOCAL_MATRIX_NAME, self._fabric_indices)
        self._parent_world_ifa = self._build_parent_indexed_array(self._sel)

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

        # Seed Fabric matrices from USD authoritatively.  ``SetWorldXformFromUsd`` /
        # ``SetLocalXformFromUsd`` are no-ops on freshly authored stages that haven't
        # been rendered yet; we instead read through the USD view (children) and
        # ``UsdGeom.XformCache`` (parents) and write via the same compose kernel that
        # ``set_world_poses`` uses.
        self._sync_fabric_from_usd_initial()

    def _sync_fabric_from_usd_initial(self) -> None:
        """Populate Fabric world+local matrices for children and parents from USD.

        Performed once during ``_initialize_fabric``.  Without this step Fabric's
        matrices are identity for stages that haven't been rendered yet, and our
        getters (which read from Fabric) would return wrong values.
        """
        # --- Children ---
        # Compose child localMatrix from USD-authored local transforms.
        # The child world matrix is NOT composed here -- it will be computed
        # by ``_recompute_world_from_local()`` at the end of this method as
        # ``child_world = child_local * parent_world``, which naturally
        # composes scales through the matrix multiplication.
        scales_wp = _to_float32_2d(self._usd_view.get_local_scales().warp)
        local_pos_ta, local_ori_ta = self._usd_view.get_local_poses()
        wp.launch(
            kernel=fabric_utils.compose_indexed_fabric_transforms,
            dim=self.count,
            inputs=[
                self._local_ifa,
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
                # Extract scale before ``Orthonormalize`` strips it from the rows.
                decomposer.SetMatrix(tf)
                s = decomposer.GetScale()
                # Check for shear/skew: after removing scale, rows should be orthogonal.
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
                            "transform. TRS decomposition (used by scale getters and world↔local "
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
            # Compose worldMatrix for parents (use a one-shot indexed array against
            # ``world_sel_rw`` keyed on the unique parent paths).
            parent_world_rw = wp.indexedfabricarray(
                fa=wp.fabricarray(self._sel, self._WORLD_MATRIX_NAME),
                indices=self._compute_fabric_indices_for(self._sel, unique_parent_paths),
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
        # the view starts with consistent state (child_world = parent_world * child_local).
        self._recompute_world_from_local()

    def _compute_fabric_indices_for(self, selection, paths: list[str]) -> wp.array:
        """Path-dict lookup helper used to build one-shot indexed arrays for a custom path set."""
        fabric_paths = selection.GetPaths()
        path_to_idx = {str(p): i for i, p in enumerate(fabric_paths)}
        indices: list[int] = []
        for path in paths:
            idx = path_to_idx.get(path)
            if idx is None:
                raise RuntimeError(f"Path '{path}' not found in Fabric selection.")
            indices.append(idx)
        return wp.array(indices, dtype=wp.int32, device=self._device)
