# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX FrameView with Fabric GPU acceleration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import warp as wp

from pxr import Gf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.app.settings_manager import SettingsManager
from isaaclab.sim.views.base_frame_view import BaseFrameView
from isaaclab.sim.views.usd_frame_view import UsdFrameView
from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.warp import fabric as fabric_utils

if TYPE_CHECKING:
    from isaaclab.sim.simulation_context import SimulationContext

logger = logging.getLogger(__name__)

# TODO: extend this to ``cuda:N`` once we wire up multi-GPU support for the view.
# Recent Kit / USDRT releases do support multi-GPU ``SelectPrims``, but the
# rest of the FabricFrameView wiring (selections, indexed arrays, etc.) still
# assumes a single device — to be tackled in a follow-up.
_fabric_supported_devices = ("cpu", "cuda", "cuda:0")


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

    World-pose, local-pose, and scale operations run on the GPU via Warp
    kernels that read and write ``omni:fabric:worldMatrix`` and
    ``omni:fabric:localMatrix`` directly.  Typical speedup vs. the
    :class:`~isaaclab.sim.views.UsdFrameView` baseline at 1024 prims is
    150-260× per call (see ``scripts/benchmarks/benchmark_view_comparison.py``).

    When Fabric is unavailable — ``/physics/fabricEnabled`` is false or the
    device is unsupported — the view transparently falls back to
    :class:`~isaaclab.sim.views.UsdFrameView` for all pose and scale
    operations.  The ``count``, ``prims``, ``prim_paths`` properties and the
    ``get_visibility`` / ``set_visibility`` methods always delegate to
    :class:`~isaaclab.sim.views.UsdFrameView`; Fabric has no equivalent fast
    path for those.

    Behavior:

    * **No write-back to USD.**  Fabric writes update only
      ``omni:fabric:worldMatrix`` / ``omni:fabric:localMatrix``; the prim's
      USD ``xformOp:*`` attributes are unchanged.  Downstream consumers that
      read the prim's USD attributes after a Fabric write will see stale
      values until the next USD-side sync.
    * **World ↔ local consistency.**  After ``set_world_poses`` (or
      ``set_scales``) the local matrix is updated so that subsequent
      ``get_local_poses`` is consistent; after ``set_local_poses`` the world
      matrix is recomputed on the next world read.  Both directions stay in
      sync without round-tripping through USD.
    * **Topology-adaptive.**  Fabric topology changes are detected on each
      access; the view rebuilds its internal mapping automatically and no
      manual refresh is required.  Steady-state overhead is negligible.

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
            device: Device for Warp arrays (``"cpu"`` or ``"cuda:0"``).
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

        if self._use_fabric and self._device not in _fabric_supported_devices:
            logger.warning(
                f"Fabric mode is not supported on device '{self._device}'. "
                "USDRT SelectPrims and Warp fabric arrays are currently "
                f"only supported on {', '.join(_fabric_supported_devices)}. "
                "Falling back to standard USD operations. This may impact performance."
            )
            self._use_fabric = False

        # Fabric state — all populated lazily in :meth:`_initialize_fabric`.
        self._fabric_initialized = False
        self._stage_id: int | None = None
        self._stage = None
        self._fabric_hierarchy = None
        # Set by ``set_local_poses``; cleared by ``_sync_world_from_local_if_dirty``.
        # Per-view (not per-stage) so concurrent views on the same stage don't clear
        # each other's flag.
        self._world_dirty: bool = False

        # Selections.
        self._trans_sel_ro = None
        self._world_sel_rw = None
        self._local_sel_rw = None

        # Index arrays (view-side indices and view→fabric mappings).  Each selection's
        # ``GetPaths()`` ordering is independent, so the view→fabric mapping is cached
        # per selection rather than shared — sharing would silently corrupt indexed
        # arrays whose selection didn't fire ``PrepareForReuse`` on the same frame.
        self._view_indices: wp.array | None = None
        self._trans_ro_fabric_indices: wp.array | None = None
        self._world_rw_fabric_indices: wp.array | None = None
        self._local_rw_fabric_indices: wp.array | None = None
        self._parent_fabric_indices: wp.array | None = None

        # Indexed fabric arrays.
        self._world_ifa_ro = None
        self._local_ifa_ro = None
        self._world_ifa_rw = None
        self._local_ifa_rw = None
        self._parent_world_ifa_ro = None

        # Sentinel passed to ``compose_indexed_fabric_transforms`` /
        # ``decompose_indexed_fabric_transforms`` for slots the caller does not want
        # written or read.  The kernels gate every per-row access on
        # ``shape[0] > 0``, so a ``(0, 0)`` array is enough — the inner dim is never
        # indexed.  One shared instance covers all "unused" slots regardless of
        # whether they would have held positions, quaternions, or scales.
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

        # If a prior set_local_poses left worldMatrix stale, propagate local → world first.
        self._sync_world_from_local_if_dirty()

        indices_wp = self._resolve_indices_wp(indices)
        positions_wp = self._to_float32_2d_or_empty(positions)
        orientations_wp = self._to_float32_2d_or_empty(orientations)

        wp.launch(
            kernel=fabric_utils.compose_indexed_fabric_transforms,
            dim=indices_wp.shape[0],
            inputs=[
                self._get_world_rw_array(),
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

        # World was just written — recompute child localMatrix from parent worldMatrix
        # so the next get_local_poses returns consistent values.
        self._sync_local_from_world(indices_wp)

    def get_world_poses(self, indices: wp.array | None = None) -> tuple[ProxyArray, ProxyArray]:
        if not self._use_fabric:
            return self._usd_view.get_world_poses(indices)

        if not self._fabric_initialized:
            self._initialize_fabric()

        # If a prior set_local_poses left worldMatrix stale, propagate local → world first.
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
                self._get_world_ro_array(),
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

        indices_wp = self._resolve_indices_wp(indices)
        translations_wp = self._to_float32_2d_or_empty(translations)
        orientations_wp = self._to_float32_2d_or_empty(orientations)

        wp.launch(
            kernel=fabric_utils.compose_indexed_fabric_transforms,
            dim=indices_wp.shape[0],
            inputs=[
                self._get_local_rw_array(),
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
        self._world_dirty = True

    def get_local_poses(self, indices: wp.array | None = None) -> tuple[ProxyArray, ProxyArray]:
        if not self._use_fabric:
            return self._usd_view.get_local_poses(indices)

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
                self._get_local_ro_array(),
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

    def set_scales(self, scales, indices=None):
        if not self._use_fabric:
            self._usd_view.set_scales(scales, indices)
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
                self._get_world_rw_array(),
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

        # World was just written — recompute child localMatrix from parent worldMatrix
        # so the next get_local_poses returns the new scale rather than the stale one.
        self._sync_local_from_world(indices_wp)

    def get_scales(self, indices=None):
        if not self._use_fabric:
            return self._usd_view.get_scales(indices)

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
                self._get_world_ro_array(),
                self._fabric_empty_2d_array_sentinel,
                self._fabric_empty_2d_array_sentinel,
                scales_wp,
                indices_wp,
            ],
            device=self._device,
        )

        if use_cached:
            wp.synchronize()
        return scales_wp

    # ------------------------------------------------------------------
    # Internal — sync helpers
    # ------------------------------------------------------------------

    def _to_float32_2d_or_empty(self, data):
        return self._fabric_empty_2d_array_sentinel if data is None else _to_float32_2d(data)

    def _sync_world_from_local_if_dirty(self) -> None:
        """If a prior local write left world matrices stale, recompute them on the fly.

        We deliberately do NOT call ``IFabricHierarchy.update_world_xforms()`` —
        in practice that re-reads USD's authored xformOps and overwrites the Fabric
        local+world matrices we just authored.  Instead we fire a Warp kernel that
        does ``child_world = parent_world * child_local`` per child, leaving the
        Fabric-side localMatrix untouched.
        """
        if not self._world_dirty:
            return
        # Refresh trans_sel_ro once, then read _local_ifa_ro and _parent_world_ifa_ro
        # directly to avoid calling PrepareForReuse twice on the same selection.
        if self._trans_sel_ro.PrepareForReuse() or self._parent_world_ifa_ro is None:
            self._rebuild_trans_ro_arrays()
        wp.launch(
            kernel=fabric_utils.update_indexed_world_matrix_from_local,
            dim=self.count,
            inputs=[
                self._local_ifa_ro,
                self._parent_world_ifa_ro,
                self._get_world_rw_array(),
                self._view_indices,
            ],
            device=self._device,
        )
        wp.synchronize()
        self._world_dirty = False

    def _sync_local_from_world(self, indices_wp: wp.array) -> None:
        """Recompute child ``localMatrix`` from (parent worldMatrix, child worldMatrix).

        Called after ``set_world_poses`` so that subsequent ``get_local_poses`` returns
        values consistent with the just-written world poses.  Fabric Hierarchy does
        not provide a built-in world → local sync, so we do it via a Warp kernel
        using the parent indexed fabric array.
        """
        # Refresh trans_sel_ro once; _world_ifa_ro and _parent_world_ifa_ro share it.
        if self._trans_sel_ro.PrepareForReuse() or self._parent_world_ifa_ro is None:
            self._rebuild_trans_ro_arrays()
        wp.launch(
            kernel=fabric_utils.update_indexed_local_matrix_from_world,
            dim=indices_wp.shape[0],
            inputs=[
                self._world_ifa_ro,
                self._parent_world_ifa_ro,
                self._get_local_rw_array(),
                indices_wp,
            ],
            device=self._device,
        )
        wp.synchronize()

    # ------------------------------------------------------------------
    # Internal — selection accessors with on-demand index rebuild
    # ------------------------------------------------------------------

    def _get_world_ro_array(self):
        if self._trans_sel_ro.PrepareForReuse():
            self._rebuild_trans_ro_arrays()
        return self._world_ifa_ro

    def _get_local_ro_array(self):
        if self._trans_sel_ro.PrepareForReuse():
            self._rebuild_trans_ro_arrays()
        return self._local_ifa_ro

    def _get_world_rw_array(self):
        if self._world_sel_rw.PrepareForReuse():
            self._world_rw_fabric_indices = self._compute_fabric_indices(self._world_sel_rw)
            self._world_ifa_rw = self._build_indexed_array(
                self._world_sel_rw, self._WORLD_MATRIX_NAME, self._world_rw_fabric_indices
            )
        return self._world_ifa_rw

    def _get_local_rw_array(self):
        if self._local_sel_rw.PrepareForReuse():
            self._local_rw_fabric_indices = self._compute_fabric_indices(self._local_sel_rw)
            self._local_ifa_rw = self._build_indexed_array(
                self._local_sel_rw, self._LOCAL_MATRIX_NAME, self._local_rw_fabric_indices
            )
        return self._local_ifa_rw

    def _get_parent_world_ro_array(self):
        # Built and refreshed alongside the trans_ro selection (parents share that selection).
        if self._parent_world_ifa_ro is None or self._trans_sel_ro.PrepareForReuse():
            self._rebuild_trans_ro_arrays()
        return self._parent_world_ifa_ro

    def _rebuild_trans_ro_arrays(self) -> None:
        """Rebuild the trans_ro indices and the three indexed arrays that depend on them.

        ``_world_ifa_ro``, ``_local_ifa_ro`` and ``_parent_world_ifa_ro`` are all
        keyed off the ``trans_sel_ro`` path ordering, so they are refreshed together.
        """
        self._trans_ro_fabric_indices = self._compute_fabric_indices(self._trans_sel_ro)
        self._world_ifa_ro = self._build_indexed_array(
            self._trans_sel_ro, self._WORLD_MATRIX_NAME, self._trans_ro_fabric_indices
        )
        self._local_ifa_ro = self._build_indexed_array(
            self._trans_sel_ro, self._LOCAL_MATRIX_NAME, self._trans_ro_fabric_indices
        )
        self._parent_world_ifa_ro = self._build_parent_indexed_array(self._trans_sel_ro)

    # ------------------------------------------------------------------
    # Internal — index computation
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
    # Internal — Fabric initialization
    # ------------------------------------------------------------------

    def _initialize_fabric(self) -> None:
        """One-time Fabric setup: hierarchy handle, attribute population, selections, indexed arrays."""
        import usdrt  # noqa: PLC0415
        from usdrt import Rt  # noqa: PLC0415

        self._stage_id = sim_utils.get_current_stage_id()
        self._stage = usdrt.Usd.Stage.Attach(self._stage_id)
        self._stage.SynchronizeToFabric()

        # Reuse (or create) a hierarchy handle for this stage via the SimulationContext.
        # The context owns the cache and clears it on stage teardown.
        fabric_id = self._stage.GetFabricId()
        self._fabric_id = fabric_id.id
        from isaaclab.sim.simulation_context import SimulationContext  # noqa: PLC0415

        sim_context = SimulationContext.instance()
        if sim_context is None:
            raise RuntimeError(
                "FabricFrameView requires an active SimulationContext. "
                "Create a SimulationContext before instantiating FabricFrameView."
            )
        self._fabric_hierarchy = sim_context.get_fabric_hierarchy(
            fabric_id.id, fabric_id, self._stage
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

        # Three persistent selections — read both, write world, write local.
        matrix = usdrt.Sdf.ValueTypeNames.Matrix4d
        ro = usdrt.Usd.Access.Read
        rw = usdrt.Usd.Access.ReadWrite
        wm_ro = (matrix, self._WORLD_MATRIX_NAME, ro)
        lm_ro = (matrix, self._LOCAL_MATRIX_NAME, ro)
        wm_rw = (matrix, self._WORLD_MATRIX_NAME, rw)
        lm_rw = (matrix, self._LOCAL_MATRIX_NAME, rw)
        self._trans_sel_ro = self._stage.SelectPrims(require_attrs=[wm_ro, lm_ro], device=self._device, want_paths=True)
        self._world_sel_rw = self._stage.SelectPrims(require_attrs=[wm_rw, lm_ro], device=self._device, want_paths=True)
        self._local_sel_rw = self._stage.SelectPrims(require_attrs=[wm_ro, lm_rw], device=self._device, want_paths=True)

        # Build the view-side indices array (just [0..count-1]) and a per-selection
        # view→fabric mapping (selections do not guarantee a shared path ordering).
        self._view_indices = wp.array(list(range(self.count)), dtype=wp.uint32, device=self._device)
        self._trans_ro_fabric_indices = self._compute_fabric_indices(self._trans_sel_ro)
        self._world_rw_fabric_indices = self._compute_fabric_indices(self._world_sel_rw)
        self._local_rw_fabric_indices = self._compute_fabric_indices(self._local_sel_rw)

        # Indexed fabric arrays per (selection × attribute).
        self._world_ifa_ro = self._build_indexed_array(
            self._trans_sel_ro, self._WORLD_MATRIX_NAME, self._trans_ro_fabric_indices
        )
        self._local_ifa_ro = self._build_indexed_array(
            self._trans_sel_ro, self._LOCAL_MATRIX_NAME, self._trans_ro_fabric_indices
        )
        self._world_ifa_rw = self._build_indexed_array(
            self._world_sel_rw, self._WORLD_MATRIX_NAME, self._world_rw_fabric_indices
        )
        self._local_ifa_rw = self._build_indexed_array(
            self._local_sel_rw, self._LOCAL_MATRIX_NAME, self._local_rw_fabric_indices
        )
        self._parent_world_ifa_ro = self._build_parent_indexed_array(self._trans_sel_ro)

        # Pre-allocated reusable output buffers (world + local + scales).
        self._fabric_positions_buf = wp.zeros((self.count, 3), dtype=wp.float32, device=self._device)
        self._fabric_orientations_buf = wp.zeros((self.count, 4), dtype=wp.float32, device=self._device)
        self._fabric_scales_buf = wp.zeros((self.count, 3), dtype=wp.float32, device=self._device)
        self._fabric_local_translations_buf = wp.zeros((self.count, 3), dtype=wp.float32, device=self._device)
        self._fabric_local_orientations_buf = wp.zeros((self.count, 4), dtype=wp.float32, device=self._device)
        self._fabric_empty_2d_array_sentinel = wp.zeros((0, 0), dtype=wp.float32, device=self._device)

        self._fabric_positions_ta = ProxyArray(self._fabric_positions_buf)
        self._fabric_orientations_ta = ProxyArray(self._fabric_orientations_buf)
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
        pos_ta, ori_ta = self._usd_view.get_world_poses()
        scales_obj = self._usd_view.get_scales()
        scales_wp = (
            scales_obj.warp
            if hasattr(scales_obj, "warp")
            else scales_obj
            if isinstance(scales_obj, wp.array)
            else self._fabric_empty_2d_array_sentinel
        )
        local_pos_ta, local_ori_ta = self._usd_view.get_local_poses()
        # Compose into child worldMatrix.
        wp.launch(
            kernel=fabric_utils.compose_indexed_fabric_transforms,
            dim=self.count,
            inputs=[
                self._world_ifa_rw,
                _to_float32_2d(pos_ta.warp),
                _to_float32_2d(ori_ta.warp),
                _to_float32_2d(scales_wp),
                False,
                False,
                False,
                self._view_indices,
            ],
            device=self._device,
        )
        # Compose into child localMatrix.  Pass the locally-authored scale so
        # that a subsequent ``_sync_world_from_local_if_dirty`` produces the
        # right world-space scale (``world = parent_world * local`` carries
        # ``local``'s scale through the multiply).
        wp.launch(
            kernel=fabric_utils.compose_indexed_fabric_transforms,
            dim=self.count,
            inputs=[
                self._local_ifa_rw,
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
            usd_stage = sim_utils.get_current_stage()
            xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            world_pos_rows: list[list[float]] = []
            world_ori_rows: list[list[float]] = []
            world_scale_rows: list[list[float]] = []
            decomposer = Gf.Transform()
            for path in unique_parent_paths:
                prim = usd_stage.GetPrimAtPath(path)
                tf = xform_cache.GetLocalToWorldTransform(prim)
                # Extract scale before ``Orthonormalize`` strips it from the rows.
                decomposer.SetMatrix(tf)
                s = decomposer.GetScale()
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
                fa=wp.fabricarray(self._world_sel_rw, self._WORLD_MATRIX_NAME),
                indices=self._compute_fabric_indices_for(self._world_sel_rw, unique_parent_paths),
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

        # The child worldMatrix above was composed with the child's *local* scale,
        # which is wrong whenever a parent has a non-unit world scale.  Mark the
        # view dirty so the next world read fires ``_sync_world_from_local_if_dirty``
        # and recomputes ``child_world = parent_world * child_local`` — that
        # multiply produces the correct world-space scale because the parent and
        # local matrices now both carry the right scale (seeded above).
        self._world_dirty = True

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
