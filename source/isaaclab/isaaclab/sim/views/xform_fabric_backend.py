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
        self._view_index_attr = f"isaaclab:view_index:{abs(id(self))}"

        # Lazy-initialized state (None until initialize() runs)
        self._view_to_fabric: wp.array | None = None
        self._default_view_indices: wp.array | None = None
        self._fabric_hierarchy = None
        self._world_selection = None
        self._local_selection = None

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
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Set up Fabric batch infrastructure for GPU-accelerated pose queries.

        Idempotent — subsequent calls after the first are no-ops.

        Ensures all prims have the required Fabric hierarchy attributes
        (``omni:fabric:localMatrix`` and ``omni:fabric:worldMatrix``) and
        creates the index mapping, selections, and pre-allocated buffers
        needed for Warp kernel launches.
        """
        if self._fabric_hierarchy is not None:
            return
        import usdrt
        from usdrt import Rt  # noqa: F401 — imported for side-effects

        stage_id = sim_utils.get_current_stage_id()
        fabric_stage = usdrt.Usd.Stage.Attach(stage_id)

        # Ensure every prim carries the view-index attribute
        for i in range(self.count):
            rt_prim = fabric_stage.GetPrimAtPath(self.prim_paths[i])
            rt_prim.CreateAttribute(self._view_index_attr, usdrt.Sdf.ValueTypeNames.UInt, custom=True)
            rt_prim.GetAttribute(self._view_index_attr).Set(i)

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

            hierarchy = (
                usdrt.hierarchy.IFabricHierarchy()
                .get_fabric_hierarchy(
                    fabric_stage.GetFabricId(),
                    fabric_stage.GetStageIdAsStageId(),
                )
            )
            hierarchy.update_world_xforms()
            hierarchy.track_local_xform_changes(True)
            hierarchy.track_world_xform_changes(True)
            FabricBackend._hierarchy_cache[stage_id] = hierarchy

        self._fabric_hierarchy = FabricBackend._hierarchy_cache[stage_id]
        self._stage_id = stage_id

        # Default view-index array (0 … count-1)
        self._default_view_indices = wp.zeros((self.count,), dtype=wp.uint32).to(self._device)
        wp.launch(
            kernel=fabric_utils.arange_k,
            dim=self.count,
            inputs=[self._default_view_indices],
            device=self._device,
        )
        wp.synchronize()

        # Resolve the Fabric device string (SelectPrims only supports cuda:0)
        fabric_device = self._device
        if self._device == "cuda":
            logger.warning("Fabric device is not specified, defaulting to 'cuda:0'.")
            fabric_device = "cuda:0"
        elif self._device.startswith("cuda:"):
            if self._device != "cuda:0":
                logger.debug(
                    f"SelectPrims only supports cuda:0. Using cuda:0 for SelectPrims "
                    f"even though simulation device is {self._device}."
                )
            fabric_device = "cuda:0"

        # Build the bidirectional view ↔ fabric index mapping
        index_selection = fabric_stage.SelectPrims(
            require_attrs=[
                (usdrt.Sdf.ValueTypeNames.UInt, self._view_index_attr, usdrt.Usd.Access.Read),
            ],
            device=fabric_device,
        )

        self._view_to_fabric = wp.zeros((self.count,), dtype=wp.uint32).to(fabric_device)
        fabric_to_view = wp.fabricarray(index_selection, self._view_index_attr)

        wp.launch(
            kernel=fabric_utils.set_view_to_fabric_array,
            dim=fabric_to_view.shape[0],
            inputs=[fabric_to_view, self._view_to_fabric],
            device=fabric_device,
        )
        wp.synchronize()

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

        self._fabric_local_translations_buffer = wp.from_torch(
            self._fabric_local_translations_torch, dtype=wp.float32
        )
        self._fabric_local_orientations_buffer = wp.from_torch(
            self._fabric_local_orientations_torch, dtype=wp.float32
        )

        # Dummy buffer for unused kernel outputs (always empty)
        self._fabric_dummy_buffer = wp.zeros((0, 3), dtype=wp.float32).to(self._device)

        self._local_selection = None
        self._fabric_local_array = None

        self._fabric_stage = fabric_stage
        self._fabric_device = fabric_device

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
        self.initialize()

        if self._stage_id in FabricBackend._dirty_stages:
            self._fabric_hierarchy.update_world_xforms()
            FabricBackend._dirty_stages.discard(self._stage_id)

        indices_wp = self._resolve_indices_wp(indices)
        count = indices_wp.shape[0]

        if positions is not None:
            positions_wp = wp.from_torch(positions)
        else:
            positions_wp = wp.zeros((0, 3), dtype=wp.float32).to(self._device)

        if orientations is not None:
            orientations_wp = wp.from_torch(orientations)
        else:
            orientations_wp = wp.zeros((0, 4), dtype=wp.float32).to(self._device)

        scales_wp = wp.zeros((0, 3), dtype=wp.float32).to(self._device)
        world_matrices = self._get_world_matrices_as_fabricarray()

        wp.launch(
            kernel=fabric_utils.compose_fabric_transformation_matrix_from_warp_arrays,
            dim=count,
            inputs=[
                world_matrices,
                positions_wp,
                orientations_wp,
                scales_wp,
                False,
                False,
                False,
                indices_wp,
                self._view_to_fabric,
            ],
            device=self._fabric_device,
        )
        wp.synchronize()

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
        self.initialize()

        indices_wp = self._resolve_indices_wp(indices)
        count = indices_wp.shape[0]

        if translations is not None:
            translations_wp = wp.from_torch(translations)
        else:
            translations_wp = wp.zeros((0, 3), dtype=wp.float32).to(self._device)

        if orientations is not None:
            orientations_wp = wp.from_torch(orientations)
        else:
            orientations_wp = wp.zeros((0, 4), dtype=wp.float32).to(self._device)

        scales_wp = wp.zeros((0, 3), dtype=wp.float32).to(self._device)
        local_matrices = self._get_local_matrices_as_fabricarray()

        wp.launch(
            kernel=fabric_utils.compose_fabric_transformation_matrix_from_warp_arrays,
            dim=count,
            inputs=[
                local_matrices,
                translations_wp,
                orientations_wp,
                scales_wp,
                False,
                False,
                False,
                indices_wp,
                self._view_to_fabric,
            ],
            device=self._fabric_device,
        )
        wp.synchronize()

        FabricBackend._dirty_stages.add(self._stage_id)

    def set_scales(self, scales: torch.Tensor, indices: Sequence[int] | None = None) -> None:
        """Write scales into the Fabric world matrix via a Warp kernel."""
        self.initialize()

        indices_wp = self._resolve_indices_wp(indices)
        count = indices_wp.shape[0]

        scales_wp = wp.from_torch(scales)
        positions_wp = wp.zeros((0, 3), dtype=wp.float32).to(self._device)
        orientations_wp = wp.zeros((0, 4), dtype=wp.float32).to(self._device)
        world_matrices = self._get_world_matrices_as_fabricarray()

        wp.launch(
            kernel=fabric_utils.compose_fabric_transformation_matrix_from_warp_arrays,
            dim=count,
            inputs=[
                world_matrices,
                positions_wp,
                orientations_wp,
                scales_wp,
                False,
                False,
                False,
                indices_wp,
                self._view_to_fabric,
            ],
            device=self._fabric_device,
        )
        wp.synchronize()

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_world_poses(self, indices: Sequence[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Read world poses from Fabric and decompose via a Warp kernel."""
        self.initialize()
        if self._stage_id in FabricBackend._dirty_stages:
            self._fabric_hierarchy.update_world_xforms()
            FabricBackend._dirty_stages.discard(self._stage_id)

        indices_wp = self._resolve_indices_wp(indices)
        count = indices_wp.shape[0]

        use_cached_buffers = indices is None or indices == slice(None)
        if use_cached_buffers:
            positions_wp = self._fabric_positions_buffer
            orientations_wp = self._fabric_orientations_buffer
            scales_wp = self._fabric_dummy_buffer
        else:
            positions_wp = wp.zeros((count, 3), dtype=wp.float32).to(self._device)
            orientations_wp = wp.zeros((count, 4), dtype=wp.float32).to(self._device)
            scales_wp = self._fabric_dummy_buffer

        world_matrices = self._get_world_matrices_as_fabricarray()

        wp.launch(
            kernel=fabric_utils.decompose_fabric_transformation_matrix_to_warp_arrays,
            dim=count,
            inputs=[
                world_matrices,
                positions_wp,
                orientations_wp,
                scales_wp,
                indices_wp,
                self._view_to_fabric,
            ],
            device=self._fabric_device,
        )

        if use_cached_buffers:
            wp.synchronize()
            return self._fabric_positions_torch, self._fabric_orientations_torch
        else:
            positions = wp.to_torch(positions_wp)
            orientations = wp.to_torch(orientations_wp)
            return positions, orientations

    def get_local_poses(self, indices: Sequence[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Read local poses from Fabric and decompose via a Warp kernel."""
        self.initialize()

        indices_wp = self._resolve_indices_wp(indices)
        count = indices_wp.shape[0]

        use_cached_buffers = indices is None or indices == slice(None)
        if use_cached_buffers:
            translations_wp = self._fabric_local_translations_buffer
            orientations_wp = self._fabric_local_orientations_buffer
            scales_wp = self._fabric_dummy_buffer
        else:
            translations_wp = wp.zeros((count, 3), dtype=wp.float32).to(self._device)
            orientations_wp = wp.zeros((count, 4), dtype=wp.float32).to(self._device)
            scales_wp = self._fabric_dummy_buffer

        local_matrices = self._get_local_matrices_as_fabricarray()

        wp.launch(
            kernel=fabric_utils.decompose_fabric_transformation_matrix_to_warp_arrays,
            dim=count,
            inputs=[
                local_matrices,
                translations_wp,
                orientations_wp,
                scales_wp,
                indices_wp,
                self._view_to_fabric,
            ],
            device=self._fabric_device,
        )

        if use_cached_buffers:
            wp.synchronize()
            return self._fabric_local_translations_torch, self._fabric_local_orientations_torch
        else:
            translations = wp.to_torch(translations_wp)
            orientations = wp.to_torch(orientations_wp)
            return translations, orientations

    def get_scales(self, indices: Sequence[int] | None = None) -> torch.Tensor:
        """Read scales from Fabric world matrices and extract via a Warp kernel."""
        self.initialize()

        indices_wp = self._resolve_indices_wp(indices)
        count = indices_wp.shape[0]

        use_cached_buffers = indices is None or indices == slice(None)
        if use_cached_buffers:
            scales_wp = self._fabric_scales_buffer
        else:
            scales_wp = wp.zeros((count, 3), dtype=wp.float32).to(self._device)

        positions_wp = self._fabric_dummy_buffer
        orientations_wp = self._fabric_dummy_buffer
        world_matrices = self._get_world_matrices_as_fabricarray()

        wp.launch(
            kernel=fabric_utils.decompose_fabric_transformation_matrix_to_warp_arrays,
            dim=count,
            inputs=[
                world_matrices,
                positions_wp,
                orientations_wp,
                scales_wp,
                indices_wp,
                self._view_to_fabric,
            ],
            device=self._fabric_device,
        )

        if use_cached_buffers:
            wp.synchronize()
            return self._fabric_scales_torch
        else:
            return wp.to_torch(scales_wp)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_world_matrices_as_fabricarray(self) -> wp.fabricarray:
        """Create a fresh :class:`wp.fabricarray` backed by ``omni:fabric:worldMatrix``.

        Rebuilding both the :class:`PrimSelection` and fabricarray each call
        ensures Fabric's journaling marks the attribute dirty for downstream
        consumers (renderers, etc.).
        """
        import usdrt

        if True:
            self._world_selection = self._fabric_stage.SelectPrims(
                require_attrs=[
                    (usdrt.Sdf.ValueTypeNames.UInt, self._view_index_attr, usdrt.Usd.Access.Read),
                    (usdrt.Sdf.ValueTypeNames.Matrix4d, self._WORLD_MATRIX_ATTR, usdrt.Usd.Access.ReadWrite),
                ],
                device=self._fabric_device,
            )
            self._fabric_array = wp.fabricarray(self._world_selection, self._WORLD_MATRIX_ATTR)
        else:
            self._world_selection.PrepareForReuse()

        return self._fabric_array

    def _get_local_matrices_as_fabricarray(self) -> wp.fabricarray:
        """Create a fresh :class:`wp.fabricarray` backed by ``omni:fabric:localMatrix``."""
        import usdrt

        if True:
            self._local_selection = self._fabric_stage.SelectPrims(
                require_attrs=[
                    (usdrt.Sdf.ValueTypeNames.UInt, self._view_index_attr, usdrt.Usd.Access.Read),
                    (usdrt.Sdf.ValueTypeNames.Matrix4d, self._LOCAL_MATRIX_ATTR, usdrt.Usd.Access.ReadWrite),
                ],
                device=self._fabric_device,
            )
            self._fabric_local_array = wp.fabricarray(self._local_selection, self._LOCAL_MATRIX_ATTR)
        else:
            self._local_selection.PrepareForReuse()

        return self._fabric_local_array

    def _resolve_indices_wp(self, indices: Sequence[int] | None) -> wp.array:
        """Convert view indices to a Warp :class:`wp.array`."""
        if indices is None or indices == slice(None):
            if self._default_view_indices is None:
                raise RuntimeError("Fabric indices are not initialized.")
            return self._default_view_indices
        indices_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)
        return wp.array(indices_list, dtype=wp.uint32).to(self._device)
