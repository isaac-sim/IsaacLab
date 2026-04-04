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

        # Resolve the Fabric device string (SelectPrims only supports cuda:0)
        if self._device.startswith("cuda"):
            if self._device == "cuda":
                logger.info("Fabric device is not specified, defaulting to 'cuda:0'.")
            elif self._device != "cuda:0":
                logger.debug(
                    "SelectPrims only supports cuda:0. Using cuda:0 even though simulation device is %s.",
                    self._device,
                )
            device = "cuda:0"
        else:
            device = self._device
        self._device = device

        import usdrt
        from usdrt import Rt  # noqa: F401 — imported for side-effects

        self._stage_id = sim_utils.get_current_stage_id()
        self._stage = usdrt.Usd.Stage.Attach(self._stage_id)

        # Reuse (or create) a hierarchy handle for this stage.
        if self._stage_id not in FabricBackend._hierarchy_cache:
            self._stage.SynchronizeToFabric()
            hierarchy = usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(
                self._stage.GetFabricId(),
                self._stage.GetStageIdAsStageId(),
            )
            hierarchy.update_world_xforms()
            hierarchy.track_local_xform_changes(True)
            hierarchy.track_world_xform_changes(True)
            FabricBackend._hierarchy_cache[self._stage_id] = hierarchy

        self._fabric_hierarchy = FabricBackend._hierarchy_cache[self._stage_id]

        matrix = usdrt.Sdf.ValueTypeNames.Matrix4d
        ro = usdrt.Usd.Access.Read
        rw = usdrt.Usd.Access.ReadWrite
        world_matrix_ro = (matrix, self._WORLD_MATRIX_ATTR, ro)
        local_matrix_ro = (matrix, self._LOCAL_MATRIX_ATTR, ro)
        world_matrix_rw = (matrix, self._WORLD_MATRIX_ATTR, rw)
        local_matrix_rw = (matrix, self._LOCAL_MATRIX_ATTR, rw)

        # Persistent selections — one per (attribute x access-mode) combination.
        # PrepareForReuse() is called before each use to detect topology changes.
        ro_ro = (world_matrix_ro, local_matrix_ro)
        ro_rw = (world_matrix_ro, local_matrix_rw)
        rw_ro = (world_matrix_rw, local_matrix_ro)

        self._trans_sel_ro = self._stage.SelectPrims(require_attrs=ro_ro, device=device, want_paths=True)
        self._world_sel_rw = self._stage.SelectPrims(require_attrs=rw_ro, device=device, want_paths=True)
        self._local_sel_rw = self._stage.SelectPrims(require_attrs=ro_rw, device=device, want_paths=True)

        # Build the view → fabric index array from PrimSelection path ordering.
        # Default view-index array [0, 1, ..., count-1] for "all prims".
        self._view_indices: wp.array = wp.array(list(range(self.count)), dtype=wp.uint32, device=self._device)
        self._fabric_indices: wp.array = self._compute_fabric_indices(self._trans_sel_ro)

        # Cached indexed fabric arrays (rebuilt when topology changes).
        self._world_ifa_ro: wp.indexedfabricarray = self._build_array(self._trans_sel_ro, self._WORLD_MATRIX_ATTR)
        self._local_ifa_ro: wp.indexedfabricarray = self._build_array(self._trans_sel_ro, self._LOCAL_MATRIX_ATTR)
        self._world_ifa_rw: wp.indexedfabricarray = self._build_array(self._world_sel_rw, self._WORLD_MATRIX_ATTR)
        self._local_ifa_rw: wp.indexedfabricarray = self._build_array(self._local_sel_rw, self._LOCAL_MATRIX_ATTR)

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

        self._decompose_transforms(self._get_world_ro_array(), fabric_indices, positions_wp, orientations_wp, dummy)

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

        self._decompose_transforms(self._get_local_ro_array(), fabric_indices, translations_wp, orientations_wp, dummy)

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
            device=self._device,
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
            device=self._device,
        )
        wp.synchronize()

    def _compute_fabric_indices(self, selection: usdrt.PrimSelection) -> wp.array:
        # Assign to each prim an index
        fabric_paths = selection.GetPaths()
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

        return wp.array(fabric_indices, dtype=wp.int32).to(self._device)

    def _ensure_fabric_indices_are_up_to_date(self, selection, force_rebuild: bool = False) -> bool:
        """Build the view index to fabric index array from PrimSelection path ordering."""

        # Rebuild indexing only when fabric topology has changed or whenever forced.
        # TODO: consider what is it cheaper, store one selection with paths and call PrepareForReuse,
        # Or each time call SelectPrims with the same paths and call PrepareForReuse?

        topology_changed = selection.PrepareForReuse()

        if topology_changed:
            logger.warning("Fabric topology changed! Rebuilding fabric indices!")

        if not (topology_changed or force_rebuild):
            return False

        self._fabric_indices = self._compute_fabric_indices(selection)
        return True

    def _build_array(self, selection: usdrt.PrimSelection, attribute_name: str) -> wp.indexedfabricarray:
        fa = wp.fabricarray(selection, attribute_name)
        return wp.indexedfabricarray(fa=fa, indices=self._fabric_indices)

    def _select_indexed(self, attr_name: str, access) -> wp.indexedfabricarray:
        """Create an indexed fabric array for a single attribute with the given access mode."""
        import usdrt

        selection = self._stage.SelectPrims(
            require_attrs=[
                (usdrt.Sdf.ValueTypeNames.Matrix4d, attr_name, access),
            ],
            device=self._device,
            want_paths=True,
        )
        fa = wp.fabricarray(selection, attr_name)
        return wp.indexedfabricarray(fa=fa, indices=self._fabric_indices)

    def _get_world_ro_array(self) -> wp.indexedfabricarray:
        # import usdrt
        # return self._select_indexed(self._WORLD_MATRIX_ATTR, usdrt.Usd.Access.Read)

        if self._trans_sel_ro.PrepareForReuse():
            self._fabric_indices = self._compute_fabric_indices(self._trans_sel_ro)
            self._world_ifa_ro = self._build_array(self._trans_sel_ro, self._WORLD_MATRIX_ATTR)
            self._local_ifa_ro = self._build_array(self._trans_sel_ro, self._LOCAL_MATRIX_ATTR)
        return self._world_ifa_ro

    def _get_local_ro_array(self) -> wp.indexedfabricarray:
        # import usdrt
        # return self._select_indexed(self._LOCAL_MATRIX_ATTR, usdrt.Usd.Access.Read)
        if self._local_sel_rw.PrepareForReuse():
            self._fabric_indices = self._compute_fabric_indices(self._local_sel_rw)
            self._world_ifa_ro = self._build_array(self._trans_sel_ro, self._WORLD_MATRIX_ATTR)
            self._local_ifa_ro = self._build_array(self._trans_sel_ro, self._LOCAL_MATRIX_ATTR)
        return self._local_ifa_rw

    def _get_world_rw_array(self) -> wp.indexedfabricarray:
        # import usdrt
        # return self._select_indexed(self._WORLD_MATRIX_ATTR, usdrt.Usd.Access.ReadWrite)
        if self._world_sel_rw.PrepareForReuse():
            self._fabric_indices = self._compute_fabric_indices(self._world_sel_rw)
            self._world_ifa_rw = self._build_array(self._world_sel_rw, self._WORLD_MATRIX_ATTR)
        return self._world_ifa_rw

    def _get_local_rw_array(self) -> wp.indexedfabricarray:
        # import usdrt
        # return self._select_indexed(self._LOCAL_MATRIX_ATTR, usdrt.Usd.Access.ReadWrite)
        if self._local_sel_rw.PrepareForReuse():
            self._fabric_indices = self._compute_fabric_indices(self._local_sel_rw)
            self._local_ifa_rw = self._build_array(self._local_sel_rw, self._LOCAL_MATRIX_ATTR)
        return self._local_ifa_rw

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

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def debug_read_fabric_matrices(self, indices: Sequence[int] | None = None) -> dict[str, list]:
        """Read world and local matrices directly from Fabric via USDRT for debugging.

        Bypasses the Warp kernel path entirely and reads raw ``Gf.Matrix4d``
        values per-prim through the USDRT prim API. Useful for verifying that
        Fabric contains the expected data after writes or population.

        Args:
            indices: Prim indices to read. Defaults to all prims.

        Returns:
            Dictionary with keys ``"prim_path"``, ``"world_matrix"``, and
            ``"local_matrix"``, each a list with one entry per queried prim.
        """
        import usdrt

        if indices is None:
            indices = list(range(self.count))

        result: dict[str, list] = {"prim_path": [], "world_matrix": [], "local_matrix": []}
        for idx in indices:
            prim_path = self.prim_paths[idx]
            rt_prim = self._stage.GetPrimAtPath(usdrt.Sdf.Path(prim_path))

            world_mat = None
            local_mat = None
            if rt_prim.IsValid():
                if rt_prim.HasAttribute(self._WORLD_MATRIX_ATTR):
                    world_mat = rt_prim.GetAttribute(self._WORLD_MATRIX_ATTR).Get()
                if rt_prim.HasAttribute(self._LOCAL_MATRIX_ATTR):
                    local_mat = rt_prim.GetAttribute(self._LOCAL_MATRIX_ATTR).Get()

            result["prim_path"].append(prim_path)
            result["world_matrix"].append(world_mat)
            result["local_matrix"].append(local_mat)

        return result

    def debug_print_fabric_state(self, indices: Sequence[int] | None = None) -> None:
        """Print Fabric matrix state, index mapping, and selection paths to stdout.

        Args:
            indices: Prim indices to print. Defaults to all prims.
        """
        if indices is None:
            indices = list(range(self.count))

        fabric_indices_np = self._fabric_indices.numpy()
        fabric_paths = self._trans_sel_ro.GetPaths()

        print(f"[Fabric Debug] stage_id={self._stage_id}  device={self._device}  count={self.count}")
        print(f"[Fabric Debug] SelectPrims returned {len(fabric_paths)} paths:")
        for fi, fp in enumerate(fabric_paths):
            print(f"  fabric_idx={fi}  path={fp}")

        print("[Fabric Debug] View → Fabric index mapping:")
        for vi in range(self.count):
            fi = int(fabric_indices_np[vi])
            print(f"  view_idx={vi}  fabric_idx={fi}  path={self.prim_paths[vi]}")

        data = self.debug_read_fabric_matrices(indices)
        print(f"[Fabric Debug] Matrices for requested indices {list(indices)}:")
        for i, path in enumerate(data["prim_path"]):
            vi = indices[i]
            fi = int(fabric_indices_np[vi])
            wm = data["world_matrix"][i]
            lm = data["local_matrix"][i]
            print(f"  [{vi}] {path} (fabric_idx={fi})")
            print(f"    world: {wm}")
            print(f"    local: {lm}")
