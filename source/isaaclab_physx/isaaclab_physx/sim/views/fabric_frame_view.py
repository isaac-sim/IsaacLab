# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX FrameView with Fabric GPU acceleration."""

from __future__ import annotations

import logging

import torch
import warp as wp

from pxr import Usd, UsdGeom

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

    After every Fabric write, :meth:`PrepareForReuse` is called on the
    ``PrimSelection`` to notify the renderer (FSD/Storm) that Fabric data
    has changed.

    All getters return ``wp.array``.  Setters accept ``wp.array``.
    """

    def __init__(
        self,
        prim_path: str,
        device: str = "cpu",
        validate_xform_ops: bool = True,
        stage: Usd.Stage | None = None,
        **kwargs,
    ):
        self._usd_view = UsdFrameView(prim_path, device=device, validate_xform_ops=validate_xform_ops, stage=stage)
        self._device = device

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

        self._prepare_for_reuse()

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

    def get_world_poses(self, indices=None):
        if not self._use_fabric:
            return self._usd_view.get_world_poses(indices)

        if not self._fabric_initialized:
            self._initialize_fabric()
        if not self._fabric_usd_sync_done:
            self._sync_fabric_from_usd_once()

        self._prepare_for_reuse()

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
    # Local poses — computed from Fabric world poses when Fabric is active
    # ------------------------------------------------------------------

    def set_local_poses(self, translations=None, orientations=None, indices=None):
        if not self._use_fabric or not self._fabric_initialized or not self._fabric_usd_sync_done:
            self._usd_view.set_local_poses(translations, orientations, indices)
            if self._use_fabric and self._fabric_initialized:
                # After writing local to USD, recompute Fabric world matrices
                self._fabric_hierarchy.update_world_xforms()
                self._prepare_for_reuse()
            return

        # Fabric path: compute child world = parent_world * local, then write to Fabric
        import torch

        indices_wp = self._resolve_indices_wp(indices)
        count = indices_wp.shape[0]
        indices_list = wp.to_torch(indices_wp).long().tolist()

        parent_pos, parent_ori = self._get_parent_world_poses(indices_list)

        if translations is not None:
            local_pos = wp.to_torch(_to_float32_2d(translations))
        else:
            local_pos = torch.zeros((count, 3), dtype=torch.float32, device=self._device)

        if orientations is not None:
            local_ori = wp.to_torch(_to_float32_2d(orientations))
        else:
            local_ori = torch.tensor([[0.0, 0.0, 0.0, 1.0]] * count, dtype=torch.float32, device=self._device)

        child_pos, child_ori = self._compose_parent_local(parent_pos, parent_ori, local_pos, local_ori)

        self.set_world_poses(
            wp.from_torch(child_pos.contiguous()),
            wp.from_torch(child_ori.contiguous()),
            indices,
        )

    def get_local_poses(self, indices=None):
        if not self._use_fabric or not self._fabric_initialized or not self._fabric_usd_sync_done:
            return self._usd_view.get_local_poses(indices)

        # Fabric path: local = inv(parent_world) * child_world

        indices_wp = self._resolve_indices_wp(indices)
        indices_list = wp.to_torch(indices_wp).long().tolist()

        child_pos_wp, child_ori_wp = self.get_world_poses(indices)
        child_pos = wp.to_torch(child_pos_wp)
        child_ori = wp.to_torch(child_ori_wp)

        parent_pos, parent_ori = self._get_parent_world_poses(indices_list)

        local_pos, local_ori = self._invert_parent_compose(parent_pos, parent_ori, child_pos, child_ori)

        return (
            wp.from_torch(local_pos.contiguous()),
            wp.from_torch(local_ori.contiguous()),
        )

    # ------------------------------------------------------------------
    # Scales — Fabric-accelerated or USD fallback
    # ------------------------------------------------------------------

    def set_scales(self, scales, indices=None):
        if not self._use_fabric:
            self._usd_view.set_scales(scales, indices)
            return

        if not self._fabric_initialized:
            self._initialize_fabric()

        self._prepare_for_reuse()

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

    def get_scales(self, indices=None):
        if not self._use_fabric:
            return self._usd_view.get_scales(indices)

        if not self._fabric_initialized:
            self._initialize_fabric()
        if not self._fabric_usd_sync_done:
            self._sync_fabric_from_usd_once()

        self._prepare_for_reuse()

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
    # Internal — PrepareForReuse (renderer notification + topology tracking)
    # ------------------------------------------------------------------

    def _prepare_for_reuse(self) -> None:
        """Call PrepareForReuse on the PrimSelection to notify the renderer.

        PrepareForReuse serves two purposes:

        1. **Renderer notification**: Tells FSD/Storm that Fabric data has
           been (or will be) modified, so the next rendered frame reflects
           the updated transforms.
        2. **Topology change detection**: Returns True when Fabric's
           internal memory layout changed (e.g., prims added/removed).
           In that case, view-to-fabric index mappings and fabricarrays
           must be rebuilt.
        """
        if self._fabric_selection is None:
            return

        topology_changed = self._fabric_selection.PrepareForReuse()
        if topology_changed:
            logger.info("Fabric topology changed — rebuilding view-to-fabric index mapping.")
            self._rebuild_fabric_arrays()

    def _rebuild_fabric_arrays(self) -> None:
        """Rebuild fabricarray and view↔fabric mappings after a topology change."""
        self._view_to_fabric = wp.zeros((self.count,), dtype=wp.uint32, device=self._fabric_device)
        self._fabric_to_view = wp.fabricarray(self._fabric_selection, self._view_index_attr)

        wp.launch(
            kernel=fabric_utils.set_view_to_fabric_array,
            dim=self._fabric_to_view.shape[0],
            inputs=[self._fabric_to_view, self._view_to_fabric],
            device=self._fabric_device,
        )
        wp.synchronize()

        self._fabric_world_matrices = wp.fabricarray(self._fabric_selection, "omni:fabric:worldMatrix")

    # ------------------------------------------------------------------
    # Internal — Local/world pose helpers
    # ------------------------------------------------------------------

    def _get_parent_world_poses(self, indices_list: list[int]) -> tuple:
        """Read parent world poses from USD for given child indices.

        Parents are not tracked in Fabric, so we read from USD XformCache.
        Returns torch tensors ``(parent_pos[N,3], parent_ori[N,4])`` on self._device.
        Orientation is ``(x, y, z, w)`` to match the convention used by FabricFrameView.
        """
        import torch

        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        stage = self._usd_view._prims[0].GetStage()

        parent_positions = []
        parent_orientations = []
        for idx in indices_list:
            child_path = self.prim_paths[idx]
            parent_path = child_path.rsplit("/", 1)[0]
            parent_prim = stage.GetPrimAtPath(parent_path)
            if parent_prim and parent_prim.IsValid():
                parent_tf = xform_cache.GetLocalToWorldTransform(parent_prim)
                parent_tf.Orthonormalize()
                t = parent_tf.ExtractTranslation()
                q = parent_tf.ExtractRotationQuat()
                img = q.GetImaginary()
                real = q.GetReal()
                parent_positions.append([float(t[0]), float(t[1]), float(t[2])])
                # (x, y, z, w) convention
                parent_orientations.append([float(img[0]), float(img[1]), float(img[2]), float(real)])
            else:
                # No parent — identity
                parent_positions.append([0.0, 0.0, 0.0])
                parent_orientations.append([0.0, 0.0, 0.0, 1.0])

        return (
            torch.tensor(parent_positions, dtype=torch.float32, device=self._device),
            torch.tensor(parent_orientations, dtype=torch.float32, device=self._device),
        )

    @staticmethod
    def _compose_parent_local(
        parent_pos: torch.Tensor,
        parent_ori: torch.Tensor,
        local_pos: torch.Tensor,
        local_ori: torch.Tensor,
    ) -> tuple:
        """Compute child_world = parent_world * local.

        Orientations are ``(x, y, z, w)``.
        Returns ``(child_world_pos, child_world_ori)``.
        """
        child_pos = parent_pos + FabricFrameView._quat_rotate(parent_ori, local_pos)
        child_ori = FabricFrameView._quat_mul(parent_ori, local_ori)
        return child_pos, child_ori

    @staticmethod
    def _invert_parent_compose(
        parent_pos: torch.Tensor,
        parent_ori: torch.Tensor,
        child_pos: torch.Tensor,
        child_ori: torch.Tensor,
    ) -> tuple:
        """Compute local = inv(parent_world) * child_world.

        Orientations are ``(x, y, z, w)``.
        Returns ``(local_pos, local_ori)``.
        """
        parent_ori_inv = FabricFrameView._quat_conjugate(parent_ori)
        local_pos = FabricFrameView._quat_rotate(parent_ori_inv, child_pos - parent_pos)
        local_ori = FabricFrameView._quat_mul(parent_ori_inv, child_ori)
        return local_pos, local_ori

    @staticmethod
    def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Quaternion multiply (x,y,z,w) convention."""
        x1, y1, z1, w1 = q1[..., 0:1], q1[..., 1:2], q1[..., 2:3], q1[..., 3:4]
        x2, y2, z2, w2 = q2[..., 0:1], q2[..., 1:2], q2[..., 2:3], q2[..., 3:4]
        import torch

        return torch.cat(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dim=-1,
        )

    @staticmethod
    def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
        """Quaternion conjugate (x,y,z,w) convention."""
        return q * q.new_tensor([-1, -1, -1, 1])

    @staticmethod
    def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Rotate vector v by quaternion q. (x,y,z,w) convention."""
        import torch

        q_xyz = q[..., :3]
        q_w = q[..., 3:4]
        t = 2.0 * torch.linalg.cross(q_xyz, v)
        return v + q_w * t + torch.linalg.cross(q_xyz, t)

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

        self.set_world_poses(positions_usd, orientations_usd)
        self.set_scales(scales_usd)

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
