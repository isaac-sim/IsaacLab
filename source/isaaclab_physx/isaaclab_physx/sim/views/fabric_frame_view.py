# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX FrameView with Fabric GPU acceleration."""

from __future__ import annotations

import logging

import torch
import warp as wp

from pxr import Usd

import isaaclab.sim as sim_utils
from isaaclab.app.settings_manager import SettingsManager
from isaaclab.sim.views.base_frame_view import BaseFrameView
from isaaclab.sim.views.usd_frame_view import UsdFrameView
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
    fallback and non-accelerated operations (local poses, visibility, scales
    when Fabric is disabled).

    When Fabric is enabled, world-pose and scale operations use GPU-accelerated
    Warp kernels operating on ``omni:fabric:worldMatrix``.  All other operations
    delegate to the internal USD view.

    All getters return ``wp.array``.  Setters accept ``wp.array``.
    """

    def __init__(
        self,
        prim_path: str,
        device: str = "cpu",
        validate_xform_ops: bool = True,
        sync_usd_on_fabric_write: bool = False,
        stage: Usd.Stage | None = None,
    ):
        self._usd_view = UsdFrameView(prim_path, device=device, validate_xform_ops=validate_xform_ops, stage=stage)
        self._device = device
        self._sync_usd_on_fabric_write = sync_usd_on_fabric_write

        settings = SettingsManager.instance()
        self._use_fabric = bool(settings.get("/physics/fabricEnabled", False))

        if self._use_fabric and self._device == "cpu":
            logger.warning(
                "Fabric mode with Warp fabric-array operations is not supported on CPU devices. "
                "Falling back to standard USD operations on the CPU. This may impact performance."
            )
            self._use_fabric = False

        if self._use_fabric and self._device not in ("cuda", "cuda:0"):
            logger.warning(
                f"Fabric mode is not supported on device '{self._device}'. "
                "USDRT SelectPrims and Warp fabric arrays only support cuda:0. "
                "Falling back to standard USD operations. This may impact performance."
            )
            self._use_fabric = False

        self._fabric_initialized = False
        self._fabric_usd_sync_done = False
        self._fabric_selection = None
        self._fabric_to_view: wp.array | None = None
        self._view_to_fabric: wp.array | None = None
        self._default_view_indices: wp.array | None = None
        self._fabric_hierarchy = None
        self._view_index_attr = f"isaaclab:view_index:{abs(hash(self))}"

    # ------------------------------------------------------------------
    # Delegated properties
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        return self._usd_view.count

    @property
    def device(self) -> str:
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

        indices_wp = self._resolve_indices_wp(indices)
        count = indices_wp.shape[0]

        dummy = wp.zeros((0, 3), dtype=wp.float32, device=self._device)
        positions_wp = _to_float32_2d(positions) if positions is not None else dummy
        orientations_wp = (
            _to_float32_2d(orientations)
            if orientations is not None
            else wp.zeros((0, 4), dtype=wp.float32, device=self._device)
        )

        wp.launch(
            kernel=fabric_utils.compose_fabric_transformation_matrix_from_warp_arrays,
            dim=count,
            inputs=[
                self._fabric_world_matrices,
                positions_wp,
                orientations_wp,
                dummy,
                False,
                False,
                False,
                indices_wp,
                self._view_to_fabric,
            ],
            device=self._fabric_device,
        )
        wp.synchronize()

        self._fabric_hierarchy.update_world_xforms()
        self._fabric_usd_sync_done = True
        if self._sync_usd_on_fabric_write:
            self._usd_view.set_world_poses(positions, orientations, indices)

    def get_world_poses(self, indices=None):
        if not self._use_fabric:
            return self._usd_view.get_world_poses(indices)

        if not self._fabric_initialized:
            self._initialize_fabric()
        if not self._fabric_usd_sync_done:
            self._sync_fabric_from_usd_once()

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
            kernel=fabric_utils.decompose_fabric_transformation_matrix_to_warp_arrays,
            dim=count,
            inputs=[
                self._fabric_world_matrices,
                positions_wp,
                orientations_wp,
                self._fabric_dummy_buffer,
                indices_wp,
                self._view_to_fabric,
            ],
            device=self._fabric_device,
        )

        if use_cached:
            wp.synchronize()
        return positions_wp, orientations_wp

    # ------------------------------------------------------------------
    # Local poses — USD fallback (Fabric only accelerates world poses)
    # ------------------------------------------------------------------

    def set_local_poses(self, translations=None, orientations=None, indices=None):
        self._usd_view.set_local_poses(translations, orientations, indices)

    def get_local_poses(self, indices=None):
        return self._usd_view.get_local_poses(indices)

    # ------------------------------------------------------------------
    # Scales — Fabric-accelerated or USD fallback
    # ------------------------------------------------------------------

    def set_scales(self, scales, indices=None):
        if not self._use_fabric:
            self._usd_view.set_scales(scales, indices)
            return

        if not self._fabric_initialized:
            self._initialize_fabric()

        indices_wp = self._resolve_indices_wp(indices)
        count = indices_wp.shape[0]

        dummy3 = wp.zeros((0, 3), dtype=wp.float32, device=self._device)
        dummy4 = wp.zeros((0, 4), dtype=wp.float32, device=self._device)
        scales_wp = _to_float32_2d(scales)

        wp.launch(
            kernel=fabric_utils.compose_fabric_transformation_matrix_from_warp_arrays,
            dim=count,
            inputs=[
                self._fabric_world_matrices,
                dummy3,
                dummy4,
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

        self._fabric_hierarchy.update_world_xforms()
        self._fabric_usd_sync_done = True
        if self._sync_usd_on_fabric_write:
            self._usd_view.set_scales(scales, indices)

    def get_scales(self, indices=None):
        if not self._use_fabric:
            return self._usd_view.get_scales(indices)

        if not self._fabric_initialized:
            self._initialize_fabric()
        if not self._fabric_usd_sync_done:
            self._sync_fabric_from_usd_once()

        indices_wp = self._resolve_indices_wp(indices)
        count = indices_wp.shape[0]

        use_cached = indices is None or indices == slice(None)
        if use_cached:
            scales_wp = self._fabric_scales_buf
        else:
            scales_wp = wp.zeros((count, 3), dtype=wp.float32, device=self._device)

        wp.launch(
            kernel=fabric_utils.decompose_fabric_transformation_matrix_to_warp_arrays,
            dim=count,
            inputs=[
                self._fabric_world_matrices,
                self._fabric_dummy_buffer,
                self._fabric_dummy_buffer,
                scales_wp,
                indices_wp,
                self._view_to_fabric,
            ],
            device=self._fabric_device,
        )

        if use_cached:
            wp.synchronize()
        return scales_wp

    # ------------------------------------------------------------------
    # Internal — Fabric initialization
    # ------------------------------------------------------------------

    def _initialize_fabric(self) -> None:
        """Initialize Fabric batch infrastructure for GPU-accelerated pose queries."""
        import usdrt  # noqa: PLC0415
        from usdrt import Rt  # noqa: PLC0415

        stage_id = sim_utils.get_current_stage_id()
        fabric_stage = usdrt.Usd.Stage.Attach(stage_id)

        for i in range(self.count):
            rt_prim = fabric_stage.GetPrimAtPath(self.prim_paths[i])
            rt_xformable = Rt.Xformable(rt_prim)

            has_attr = (
                rt_xformable.HasFabricHierarchyWorldMatrixAttr()
                if hasattr(rt_xformable, "HasFabricHierarchyWorldMatrixAttr")
                else False
            )
            if not has_attr:
                rt_xformable.CreateFabricHierarchyWorldMatrixAttr()

            rt_xformable.SetWorldXformFromUsd()

            rt_prim.CreateAttribute(self._view_index_attr, usdrt.Sdf.ValueTypeNames.UInt, custom=True)
            rt_prim.GetAttribute(self._view_index_attr).Set(i)

        self._fabric_hierarchy = usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(
            fabric_stage.GetFabricId(), fabric_stage.GetStageIdAsStageId()
        )
        self._fabric_hierarchy.update_world_xforms()

        self._default_view_indices = wp.zeros((self.count,), dtype=wp.uint32, device=self._device)
        wp.launch(
            kernel=fabric_utils.arange_k, dim=self.count, inputs=[self._default_view_indices], device=self._device
        )
        wp.synchronize()

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

        self._fabric_selection = fabric_stage.SelectPrims(
            require_attrs=[
                (usdrt.Sdf.ValueTypeNames.UInt, self._view_index_attr, usdrt.Usd.Access.Read),
                (usdrt.Sdf.ValueTypeNames.Matrix4d, "omni:fabric:worldMatrix", usdrt.Usd.Access.ReadWrite),
            ],
            device=fabric_device,
        )

        self._view_to_fabric = wp.zeros((self.count,), dtype=wp.uint32, device=fabric_device)
        self._fabric_to_view = wp.fabricarray(self._fabric_selection, self._view_index_attr)

        wp.launch(
            kernel=fabric_utils.set_view_to_fabric_array,
            dim=self._fabric_to_view.shape[0],
            inputs=[self._fabric_to_view, self._view_to_fabric],
            device=fabric_device,
        )
        wp.synchronize()

        self._fabric_positions_buf = wp.zeros((self.count, 3), dtype=wp.float32, device=self._device)
        self._fabric_orientations_buf = wp.zeros((self.count, 4), dtype=wp.float32, device=self._device)
        self._fabric_scales_buf = wp.zeros((self.count, 3), dtype=wp.float32, device=self._device)
        self._fabric_dummy_buffer = wp.zeros((0, 3), dtype=wp.float32, device=self._device)
        self._fabric_world_matrices = wp.fabricarray(self._fabric_selection, "omni:fabric:worldMatrix")
        self._fabric_stage = fabric_stage
        self._fabric_device = fabric_device

        self._fabric_initialized = True
        self._fabric_usd_sync_done = False

    def _sync_fabric_from_usd_once(self) -> None:
        """Sync Fabric world matrices from USD once, on the first read."""
        if not self._fabric_initialized:
            self._initialize_fabric()

        positions_usd, orientations_usd = self._usd_view.get_world_poses()
        scales_usd = self._usd_view.get_scales()

        prev_sync = self._sync_usd_on_fabric_write
        self._sync_usd_on_fabric_write = False
        self.set_world_poses(positions_usd, orientations_usd)
        self.set_scales(scales_usd)
        self._sync_usd_on_fabric_write = prev_sync

        self._fabric_usd_sync_done = True

    def _resolve_indices_wp(self, indices: wp.array | None) -> wp.array:
        """Resolve view indices as a Warp uint32 array."""
        if indices is None or indices == slice(None):
            if self._default_view_indices is None:
                raise RuntimeError("Fabric indices are not initialized.")
            return self._default_view_indices
        if indices.dtype != wp.uint32:
            return wp.array(indices.numpy().astype("uint32"), dtype=wp.uint32, device=self._device)
        return indices
