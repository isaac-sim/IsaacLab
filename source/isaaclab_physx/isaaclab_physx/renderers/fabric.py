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
from pxr import Usd, UsdGeom, UsdUtils

from isaaclab.scene_data import SceneDataFormat, SceneDataProvider
from isaaclab.sim import BackendCfg
from isaaclab.utils import configclass
from isaaclab.utils.buffers import TimestampedBuffer


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
    """Own one stage's native Fabric handles and shared transform/geometry bindings.

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
        self._transforms_timestamp = -1
        self._geometry_bindings = None

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
        timestamp = provider.backend.transforms_timestamp
        if self._selection is not None and (changed or self._transforms_timestamp != timestamp):
            self._write_selection.PrepareForReuse()
            device = self._scales.device
            wp.synchronize_stream(device)
            if not self.hierarchy.update_world_xforms_gpu(not changed and self._transforms_timestamp != -1):
                raise RuntimeError("Fabric GPU transform hierarchy update failed.")
            wp.synchronize_device(device)
        self._transforms_timestamp = timestamp

    def update_geometries(self, provider: SceneDataProvider, frame: int) -> None:
        """Request world-space visual vertices [m] directly into due Fabric destinations."""
        # The native transform request refreshes the whole Fabric stage, including geometry.
        if SceneDataFormat.FabricMatrix44 in provider.backend.native_transform_formats:
            return
        batches = provider.backend.get_geometry_batches()
        if self._geometry_bindings is None:
            self._bind_geometries(provider, tuple(path for _, ranges in batches for path in ranges))
        timestamp = provider.backend.geometry_timestamp
        for index, (selections, frequency, cached, offsets, frame_last_update) in enumerate(self._geometry_bindings):
            selection, write_selection = selections
            changed = selection.PrepareForReuse()
            if (
                cached.data is not None
                and not changed
                and (timestamp == cached.timestamp or frequency > 1 and frame - frame_last_update < frequency)
            ):
                continue
            write_selection.PrepareForReuse()
            output = cached.data
            if output is None or changed:
                cached.data = None  # Retry the binding too if conversion fails after the selection changes.
                indices = wp.fabricarray(selection, f"isaaclab:geometryIndex:group{index}").numpy()
                rows = {int(value): row for row, value in enumerate(indices)}
                offsets = {path: rows[row] for row, path in enumerate(offsets)}
                output = SceneDataFormat.FabricPoints()
                output.points = wp.fabricarrayarray(data=write_selection, attrib="points", dtype=wp.vec3f)
            provider.get_geometry_points(output=output, offsets=offsets)
            cached.data, cached.timestamp = output, provider.backend.geometry_timestamp
            self._geometry_bindings[index] = (selections, frequency, cached, offsets, frame)

    def _bind_geometries(self, provider: SceneDataProvider, paths: tuple[str, ...]) -> None:
        """Bind declared destinations by device and cadence; SDP owns all data movement."""
        # Foreign physics publishes world points. Author the sink once so Kit's USD refresh agrees.
        groups = {}
        for path in paths:
            geometry = UsdGeom.Xformable(provider.usd_stage.GetPrimAtPath(path))
            geometry.ClearXformOpOrder()
            geometry.SetResetXformStack(True)
            prim_type = geometry.GetPrim().GetTypeName()
            if prim_type == "Mesh":
                device, frequency = self.device, 1
            elif prim_type in {"BasisCurves", "Points"}:
                # RTX Hydra reads these points from CPU Fabric (BasisCurves: NVBug 6502662).
                device = "cpu"
                frequency = geometry.GetPrim().GetAttribute("isaaclab:pointsUpdateFrequency").Get() or 1
            else:
                raise TypeError(f"Unsupported Fabric point destination {path}: {prim_type}.")
            groups.setdefault((device, frequency), []).append(path)
        self.stage.SynchronizeToFabric()
        self._geometry_bindings = []
        for index, ((device, frequency), group_paths) in enumerate(groups.items()):
            tag = f"isaaclab:geometryIndex:group{index}"
            for row, path in enumerate(group_paths):
                prim = self.stage.GetPrimAtPath(path)
                usdrt.Rt.Xformable(prim).SetWorldXformFromUsd()
                prim.CreateAttribute(tag, usdrt.Sdf.ValueTypeNames.Int, custom=True).Set(row)
            attrs = [
                (usdrt.Sdf.ValueTypeNames.Point3fArray, "points", usdrt.Usd.Access.Read),
                (usdrt.Sdf.ValueTypeNames.Int, tag, usdrt.Usd.Access.Read),
            ]
            selection = self.stage.SelectPrims(require_attrs=attrs, device=device)
            write_selection = self.stage.SelectPrims(
                require_attrs=[(*attrs[0][:2], usdrt.Usd.Access.ReadWrite), attrs[1]],
                device=device,
            )
            offsets = dict.fromkeys(group_paths, 0)
            self._geometry_bindings.append(
                ((selection, write_selection), frequency, TimestampedBuffer(), offsets, -frequency)
            )

    def close(self) -> None:
        """Release stage-bound selections and borrowed SDP buffers."""
        self.transforms = self._selection = self._write_selection = None
        self._mapping = self._scales = self.hierarchy = self.stage = None
        self._geometry_bindings = None


@configclass
class FabricBackendCfg(BackendCfg):
    """Native Fabric identity; the stage is borrowed from the active simulation."""

    class_type: type = FabricBackend
    stage: Usd.Stage = field(kw_only=True, metadata={"copy": False})
    device: str = field(kw_only=True)
