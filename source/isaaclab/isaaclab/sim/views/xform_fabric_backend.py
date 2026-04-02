# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
import warp as wp

from pxr import Usd

import isaaclab.sim as sim_utils
from isaaclab.utils.warp import fabric as fabric_utils

logger = logging.getLogger(__name__)


class FabricBackend:
    """Fabric-based transform backend for :class:`XformPrimView`.

    Uses NVIDIA's Fabric API with Warp GPU kernels for high-performance batch
    transform operations.

    Selected primitives based on attributes such as local and world matrices.
    (all prims in selection, could be in different buckets):
    ┌─────────┬───────────────┬───────────┐
    │ Fab Idx │ Prim Path     │ attribute │
    ├─────────┼───────────────┼───────────┤
    │    0    │ /World/Light  │  [ ... ]  │
    │    1    │ /World/Cam_2  │  [ ... ]  │
    │    2    │ /World/Ground │  [ ... ]  │
    │    3    │ /World/Cam_0  │  [ ... ]  │
    │    4    │ /World/Table  │  [ ... ]  │
    │    5    │ /World/Cam_1  │  [ ... ]  │
    │    6    │ /World/Robot  │  [ ... ]  │
    └─────────┴───────────────┴───────────┘

    Example view of 3 prims, order of the paths defines order of indices
    ┌──────────┬──────────────┐
    │ View Idx │ Prim Path    │
    ├──────────┼──────────────┤
    │    0     │ /World/Cam_0 │
    │    1     │ /World/Cam_1 │
    │    2     │ /World/Cam_2 │
    └──────────┴──────────────┘

    Mapping from view indices to fabric indices happens through a fabric index array:
    ┌──────────┬─────────┐
    │ View Idx │ Fab Idx │
    ├──────────┼─────────┤
    │    0     │    3    │
    │    1     │    5    │
    │    2     │    1    │
    └──────────┴─────────┘

    If topology of the fabric changes, then all fabric indices need to be rebuilt.
    """

    _WORLD_MATRIX_ATTR = "omni:fabric:worldMatrix"
    _LOCAL_MATRIX_ATTR = "omni:fabric:localMatrix"

    _hierarchy_cache: dict[int, object] = {}
    _dirty_stages: set[int] = set()

    def __init__(
        self,
        prims: list[Usd.Prim],
        device: str,
    ):
        self._prims = prims
        self._device = device

        # Lazy-initialized state (None until __init__ body completes)
        self._view_indices: wp.array | None = None
        self._fabric_indices: wp.array | None = None

        # Resolve the Fabric device string (SelectPrims only supports cuda:0)
        if self._device.startswith("cuda"):
            if self._device == "cuda":
                logger.warning("Fabric device is not specified, defaulting to 'cuda:0'.")
            elif self._device != "cuda:0":
                logger.debug(
                    "SelectPrims only supports cuda:0. Using cuda:0 even though simulation device is %s.",
                    self._device,
                )
            fabric_device = "cuda:0"
        else:
            fabric_device = self._device
        self._fabric_device = fabric_device

        import usdrt
        from usdrt import Rt  # noqa: F401 — imported for side-effects

        stage_id = sim_utils.get_current_stage_id()
        fabric_stage = usdrt.Usd.Stage.Attach(stage_id)
        self._fabric_stage = fabric_stage

        # Reuse (or create) a hierarchy handle for this stage.
        if stage_id not in FabricBackend._hierarchy_cache:
            pop = usdrt.population.IUtils()
            pop.set_enable_usd_notice_handling(
                fabric_stage.GetStageIdAsStageId(),
                fabric_stage.GetFabricId(),
                True,
            )
            pop.populate_from_usd(
                fabric_stage.GetStageReaderWriterId(),
                fabric_stage.GetStageIdAsStageId(),
                usdrt.Sdf.Path("/"),
                0,
            )
            pop.apply_pending_usd_updates(
                fabric_stage.GetStageIdAsStageId(),
                fabric_stage.GetStageReaderWriterId(),
                0,
            )

            hierarchy = usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(
                fabric_stage.GetFabricId(),
                fabric_stage.GetStageIdAsStageId(),
            )
            hierarchy.update_world_xforms()
            hierarchy.track_local_xform_changes(True)
            hierarchy.track_world_xform_changes(True)
            FabricBackend._hierarchy_cache[stage_id] = hierarchy

        self._fabric_hierarchy = FabricBackend._hierarchy_cache[stage_id]
        self._stage_id = stage_id

        # Index selection for all primitives populated by the hierarchy, used for tracking topology changes.
        self._index_selection = fabric_stage.SelectPrims(
            require_attrs=[
                (usdrt.Sdf.ValueTypeNames.Matrix4d, self._LOCAL_MATRIX_ATTR, usdrt.Usd.Access.Read),
                (usdrt.Sdf.ValueTypeNames.Matrix4d, self._WORLD_MATRIX_ATTR, usdrt.Usd.Access.Read),
            ],
            device=fabric_device,
            want_paths=True,
        )

        # Build the view → fabric index array from PrimSelection path ordering.
        # Default view-index array [0, 1, ..., count-1] for "all prims".
        self._rebuild_view_to_fabric_index_mapping(force_rebuild=True)

        # Pre-allocate reusable output buffers (world poses)
        self._fabric_positions_torch = torch.zeros((self.count, 3), dtype=torch.float32, device=self._device)
        self._fabric_orientations_torch = torch.zeros((self.count, 4), dtype=torch.float32, device=self._device)
        self._fabric_scales_torch = torch.zeros((self.count, 3), dtype=torch.float32, device=self._device)

        self._fabric_positions_buffer = wp.from_torch(self._fabric_positions_torch, dtype=wp.float32)
        self._fabric_orientations_buffer = wp.from_torch(self._fabric_orientations_torch, dtype=wp.float32)
        self._fabric_scales_buffer = wp.from_torch(self._fabric_scales_torch, dtype=wp.float32)

        # Pre-allocate reusable output buffers (local poses)
        self._fabric_local_translations_torch = torch.zeros((self.count, 3), dtype=torch.float32, device=self._device)
        self._fabric_local_orientations_torch = torch.zeros((self.count, 4), dtype=torch.float32, device=self._device)

        self._fabric_local_translations_buffer = wp.from_torch(self._fabric_local_translations_torch, dtype=wp.float32)
        self._fabric_local_orientations_buffer = wp.from_torch(self._fabric_local_orientations_torch, dtype=wp.float32)

        # Dummy buffer for unused kernel outputs (always empty)
        self._fabric_dummy_buffer = wp.zeros((0, 3), dtype=wp.float32).to(self._device)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of prims managed by this backend."""
        return len(self._prims)

    @property
    def prim_paths(self) -> list[str]:
        """Prim path strings (lazily cached)."""
        if not hasattr(self, "_prim_paths_cache"):
            self._prim_paths_cache = [p.GetPath().pathString for p in self._prims]
        return self._prim_paths_cache

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------

    def set_world_poses(
        self,
        positions: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        indices: Sequence[int] | None = None,
    ) -> None:
        """Write world poses to Fabric ``omni:fabric:worldMatrix`` via a Warp kernel."""

        # if local transforms were set, we need to update the world transforms
        if self._stage_id in FabricBackend._dirty_stages:
            self._fabric_hierarchy.update_world_xforms()
            FabricBackend._dirty_stages.discard(self._stage_id)

        fabric_indices = self._convert_view_to_fabric_indices(indices)
        self._compose_transforms(
            self._get_world_rw_array(), fabric_indices, positions=positions, orientations=orientations
        )

    def set_local_poses(
        self,
        translations: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        indices: Sequence[int] | None = None,
    ) -> None:
        """Write local poses to Fabric ``omni:fabric:localMatrix`` via a Warp kernel.

        After composing the local matrix the method re-registers it through
        :pyobj:`IFabricHierarchy` and marks world transforms as dirty so that a
        subsequent read will propagate the change.
        """
        fabric_indices = self._convert_view_to_fabric_indices(indices)
        self._compose_transforms(
            self._get_local_rw_array(), fabric_indices, positions=translations, orientations=orientations
        )

        FabricBackend._dirty_stages.add(self._stage_id)

    def set_scales(self, scales: torch.Tensor, indices: Sequence[int] | None = None) -> None:
        """Write scales into the Fabric world matrix via a Warp kernel."""
        fabric_indices = self._convert_view_to_fabric_indices(indices)
        self._compose_transforms(self._get_world_rw_array(), fabric_indices, scales=scales)

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_world_poses(self, indices: Sequence[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Read world poses from Fabric and decompose via a Warp kernel."""
        if self._stage_id in FabricBackend._dirty_stages:
            self._fabric_hierarchy.update_world_xforms()
            FabricBackend._dirty_stages.discard(self._stage_id)

        fabric_indices = self._convert_view_to_fabric_indices(indices)
        count = fabric_indices.shape[0]
        dummy = self._fabric_dummy_buffer

        use_cached_buffers = indices is None or indices == slice(None)
        if use_cached_buffers:
            positions_wp = self._fabric_positions_buffer
            orientations_wp = self._fabric_orientations_buffer
        else:
            positions_wp = wp.zeros((count, 3), dtype=wp.float32).to(self._device)
            orientations_wp = wp.zeros((count, 4), dtype=wp.float32).to(self._device)

        self._decompose_transforms(
            self._get_world_ro_array(), fabric_indices, positions_wp, orientations_wp, dummy
        )

        if use_cached_buffers:
            return self._fabric_positions_torch, self._fabric_orientations_torch
        return wp.to_torch(positions_wp), wp.to_torch(orientations_wp)

    def get_local_poses(self, indices: Sequence[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Read local poses from Fabric and decompose via a Warp kernel."""
        fabric_indices = self._convert_view_to_fabric_indices(indices)
        count = fabric_indices.shape[0]
        dummy = self._fabric_dummy_buffer

        use_cached_buffers = indices is None or indices == slice(None)
        if use_cached_buffers:
            translations_wp = self._fabric_local_translations_buffer
            orientations_wp = self._fabric_local_orientations_buffer
        else:
            translations_wp = wp.zeros((count, 3), dtype=wp.float32).to(self._device)
            orientations_wp = wp.zeros((count, 4), dtype=wp.float32).to(self._device)

        self._decompose_transforms(
            self._get_local_ro_array(), fabric_indices, translations_wp, orientations_wp, dummy
        )

        if use_cached_buffers:
            return self._fabric_local_translations_torch, self._fabric_local_orientations_torch
        return wp.to_torch(translations_wp), wp.to_torch(orientations_wp)

    def get_scales(self, indices: Sequence[int] | None = None) -> torch.Tensor:
        """Read scales from Fabric world matrices and extract via a Warp kernel."""
        fabric_indices = self._convert_view_to_fabric_indices(indices)
        count = fabric_indices.shape[0]
        dummy = self._fabric_dummy_buffer

        use_cached_buffers = indices is None or indices == slice(None)
        if use_cached_buffers:
            scales_wp = self._fabric_scales_buffer
        else:
            scales_wp = wp.zeros((count, 3), dtype=wp.float32).to(self._device)

        self._decompose_transforms(self._get_world_ro_array(), fabric_indices, dummy, dummy, scales_wp)

        if use_cached_buffers:
            return self._fabric_scales_torch
        return wp.to_torch(scales_wp)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compose_transforms(
        self,
        matrices: wp.indexedfabricarray,
        indices_wp: wp.array,
        positions: torch.Tensor | None = None,
        orientations: torch.Tensor | None = None,
        scales: torch.Tensor | None = None,
    ) -> None:
        """Launch the compose kernel to write transform components into Fabric matrices.

        Converts non-``None`` torch tensors to Warp arrays and substitutes a
        pre-allocated zero-length dummy for omitted components so the kernel
        leaves existing values untouched.
        """
        dummy = self._fabric_dummy_buffer
        positions_wp = wp.from_torch(positions) if positions is not None else dummy
        orientations_wp = wp.from_torch(orientations) if orientations is not None else dummy
        scales_wp = wp.from_torch(scales) if scales is not None else dummy

        wp.launch(
            kernel=fabric_utils.compose_indexed_fabric_transforms,
            dim=indices_wp.shape[0],
            inputs=[
                matrices,
                positions_wp,
                orientations_wp,
                scales_wp,
                False,
                False,
                False,
                indices_wp,
            ],
            device=self._fabric_device,
        )
        wp.synchronize()

    def _decompose_transforms(
        self,
        matrices: wp.indexedfabricarray,
        indices_wp: wp.array,
        positions_wp: wp.array,
        orientations_wp: wp.array,
        scales_wp: wp.array,
    ) -> None:
        """Launch the decompose kernel to read transform components from Fabric matrices."""
        wp.launch(
            kernel=fabric_utils.decompose_indexed_fabric_transforms,
            dim=indices_wp.shape[0],
            inputs=[matrices, positions_wp, orientations_wp, scales_wp, indices_wp],
            device=self._fabric_device,
        )
        wp.synchronize()

    def _rebuild_view_to_fabric_index_mapping(self, force_rebuild: bool = False) -> None:
        """Build the view index to fabric index array from PrimSelection path ordering."""

        # Rebuild indexing only when fabric topology has changed or whenever forced.
        topology_changed = self._index_selection.PrepareForReuse()
        if not (topology_changed or force_rebuild):
            return

        self._view_indices = wp.array(list(range(self.count)), dtype=wp.uint32, device=self._device)

        # Assign to each prim an index
        fabric_paths = self._index_selection.GetPaths()
        path_to_fabric_idx: dict[str, int] = {str(p): i for i, p in enumerate(fabric_paths)}

        # Look up the index for each prim observed by this view
        fabric_indices: list[int] = []
        for prim_path in self.prim_paths:
            fabric_idx = path_to_fabric_idx.get(prim_path)
            if fabric_idx is None:
                raise RuntimeError(
                    f"Prim '{prim_path}' not found in Fabric selection. Ensure the hierarchy has been populated."
                )
            fabric_indices.append(fabric_idx)

        self._fabric_indices = wp.array(fabric_indices, dtype=wp.int32).to(self._fabric_device)

    def _select_indexed(self, attr_name: str, access) -> wp.indexedfabricarray:
        """Create an indexed fabric array for a single attribute with the given access mode."""
        import usdrt
        selection = self._fabric_stage.SelectPrims(
            require_attrs=[
                (usdrt.Sdf.ValueTypeNames.Matrix4d, attr_name, access),
            ],
            device=self._fabric_device,
        )
        fa = wp.fabricarray(selection, attr_name)
        return wp.indexedfabricarray(fa=fa, indices=self._fabric_indices)

    def _get_world_ro_array(self) -> wp.indexedfabricarray:
        import usdrt
        return self._select_indexed(self._WORLD_MATRIX_ATTR, usdrt.Usd.Access.Read)

    def _get_world_rw_array(self) -> wp.indexedfabricarray:
        import usdrt
        self._rebuild_view_to_fabric_index_mapping()
        return self._select_indexed(self._WORLD_MATRIX_ATTR, usdrt.Usd.Access.ReadWrite)

    def _get_local_ro_array(self) -> wp.indexedfabricarray:
        import usdrt
        return self._select_indexed(self._LOCAL_MATRIX_ATTR, usdrt.Usd.Access.Read)

    def _get_local_rw_array(self) -> wp.indexedfabricarray:
        import usdrt
        self._rebuild_view_to_fabric_index_mapping()
        return self._select_indexed(self._LOCAL_MATRIX_ATTR, usdrt.Usd.Access.ReadWrite)

    def _convert_view_to_fabric_indices(self, indices: Sequence[int] | None) -> wp.array:
        """Convert requested view indices to fabric indices.

        Args:
            indices: Requested path indices. If None, then all indices are used.

        Returns:
            A warp array of fabric indices.
        """
        if indices is None or indices == slice(None):
            if self._view_indices is None:
                raise RuntimeError("Fabric indices are not initialized.")
            return self._view_indices
        indices_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)
        return wp.array(indices_list, dtype=wp.uint32).to(self._device)
