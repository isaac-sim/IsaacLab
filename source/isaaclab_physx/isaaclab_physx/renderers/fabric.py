# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Simulation-owned Fabric resource shared by Kit and Isaac RTX."""

from __future__ import annotations

from dataclasses import field

import warp as wp

import usdrt
import usdrt.hierarchy
from pxr import Usd, UsdUtils

from isaaclab.scene_data import SceneDataFormat, SceneDataProvider
from isaaclab.sim import BackendCfg
from isaaclab.utils import configclass


@wp.kernel(enable_backward=False)
def _capture_scales(
    matrices: wp.fabricarray(dtype=wp.mat44d),
    indices: wp.fabricarray(dtype=wp.int32),
    scales: wp.array(dtype=wp.vec3f),
):
    i = wp.tid()
    matrix = wp.mat44f(matrices[i])
    scales[indices[i]] = wp.vec3f(
        wp.length(wp.vec3f(matrix[0, 0], matrix[0, 1], matrix[0, 2])),
        wp.length(wp.vec3f(matrix[1, 0], matrix[1, 1], matrix[1, 2])),
        wp.length(wp.vec3f(matrix[2, 0], matrix[2, 1], matrix[2, 2])),
    )


class FabricBackend:
    """Own one stage's native Fabric handles and shared transform bindings.

    Consumers supply the simulation's SDP explicitly. Its producer layout stays fixed for the
    binding's lifetime; it is not part of the native stage/device identity.
    """

    def __init__(self, cfg: FabricBackendCfg):
        self.device = cfg.device
        self.stage = usdrt.Usd.Stage.Attach(UsdUtils.StageCache.Get().GetId(cfg.stage).ToLongInt())
        self.hierarchy = usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(
            self.stage.GetFabricId(), self.stage.GetStageIdAsStageId()
        )
        self.transforms: SceneDataFormat.FabricMatrix44 | None = None
        self._selection = self._write_selection = None
        self._mapping = self._scales = None
        self._version = -1

    def bind_transforms(self, provider: SceneDataProvider) -> None:
        """Bind the initialized simulation's rigid destinations once; native Fabric needs no conversion binding."""
        if self.transforms is not None:
            return
        if SceneDataFormat.FabricMatrix44 in provider.backend.native_transform_formats:
            self.transforms = SceneDataFormat.FabricMatrix44()
            return

        stage = self.stage
        stage.SynchronizeToFabric()
        self.hierarchy.update_world_xforms()
        for index, path in enumerate(provider.backend.transform_paths):
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.HasAPI("PhysicsRigidBodyAPI"):
                continue
            prim.CreateAttribute("isaaclab:transformIndex", usdrt.Sdf.ValueTypeNames.Int, custom=True).Set(index)
            # Physics publishes absolute body poses; only visual descendants inherit them.
            self.hierarchy.set_reset_xform_stack(prim.GetPath().fabricPath, True)
        attrs = [
            (usdrt.Sdf.ValueTypeNames.Matrix4d, "omni:fabric:worldMatrix", usdrt.Usd.Access.Read),
            (usdrt.Sdf.ValueTypeNames.Int, "isaaclab:transformIndex", usdrt.Usd.Access.Read),
            (usdrt.Sdf.ValueTypeNames.Matrix4d, "omni:fabric:localMatrix", usdrt.Usd.Access.Read),
        ]
        self._selection = stage.SelectPrims(require_attrs=attrs, device=self.device)
        self._write_selection = stage.SelectPrims(
            require_attrs=[*attrs[:-1], (*attrs[-1][:2], usdrt.Usd.Access.ReadWrite)], device=self.device
        )
        self._scales = wp.empty(provider.transform_count, dtype=wp.vec3f, device=self.device)
        self.transforms = SceneDataFormat.FabricMatrix44()

    def update_transforms(self, provider: SceneDataProvider) -> None:
        """Request SDP transforms and propagate converted body matrices to visual descendants."""
        self.bind_transforms(provider)
        changed = self._selection is not None and self._selection.PrepareForReuse()
        if self._selection is not None and (changed or self.transforms.matrices is None):
            self._write_selection.PrepareForReuse()
            self._mapping = wp.fabricarray(self._selection, "isaaclab:transformIndex")
            if self.transforms.matrices is None:
                wp.launch(
                    _capture_scales,
                    dim=len(self._mapping),
                    inputs=[wp.fabricarray(self._selection, "omni:fabric:worldMatrix"), self._mapping],
                    outputs=[self._scales],
                    device=self._scales.device,
                )
            self.transforms = SceneDataFormat.FabricMatrix44()
            self.transforms.matrices = wp.fabricarray(self._write_selection, "omni:fabric:localMatrix")
        provider.get_transforms(self.transforms, self._mapping, scales=self._scales)
        version = provider.backend.transforms_version
        if self._selection is not None and (changed or self._version != version):
            self._write_selection.PrepareForReuse()
            device = self._scales.device
            wp.synchronize_stream(device)
            if not self.hierarchy.update_world_xforms_gpu(not changed and self._version != -1):
                raise RuntimeError("Fabric GPU transform hierarchy update failed.")
            wp.synchronize_device(device)
        self._version = version

    def close(self) -> None:
        """Release stage-bound selections and borrowed SDP buffers."""
        self.transforms = self._selection = self._write_selection = None
        self._mapping = self._scales = self.hierarchy = self.stage = None


@configclass
class FabricBackendCfg(BackendCfg):
    """Native Fabric identity; the stage is borrowed from the active simulation."""

    class_type: type = FabricBackend
    stage: Usd.Stage = field(kw_only=True, metadata={"copy": False})
    device: str = field(kw_only=True)
