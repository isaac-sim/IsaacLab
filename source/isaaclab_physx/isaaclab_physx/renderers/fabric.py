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
from pxr import Sdf, Usd, UsdGeom, UsdUtils, Vt

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


@wp.kernel(enable_backward=False)
def _write_geometry(
    destinations: wp.fabricarrayarray(dtype=wp.vec3f),
    indices: wp.fabricarray(dtype=wp.int32),
    sources: wp.array(dtype=SceneDataFormat.Points),
):
    i, j = wp.tid()
    index = int(indices[i])
    source = sources[index].points
    if j < source.shape[0]:
        destinations[i][j] = source[j]


@wp.kernel(enable_backward=False)
def _pack_geometry(sources: wp.array(dtype=SceneDataFormat.Points), targets: wp.array(dtype=SceneDataFormat.Points)):
    i, j = wp.tid()
    target = targets[i].points
    if j < target.shape[0]:
        target[j] = sources[i].points[j]


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
        self._version = -1
        self._geometry_points = None
        self._geometry_sources = self._geometry_selection = self._curve_selection = None
        self._host_geometry = None
        self._point_clouds = []
        self._geometry_version = -1

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

    def update_geometries(self, provider: SceneDataProvider, frame: int) -> None:
        """Write SDP's world-space visual vertices [m], preserving USD point-cloud render cadence."""
        if SceneDataFormat.FabricPoints in provider.backend.native_geometry_formats:
            provider.get_geometry_points(output_format=SceneDataFormat.FabricPoints)
            return
        points = provider.get_geometry_points()
        version = provider.backend.geometry_version
        if self._geometry_points is None:
            self._bind_geometries(provider, points)
        if not points:
            return
        changed = version != self._geometry_version
        due = [
            cloud
            for cloud in self._point_clouds
            if cloud[4] != version and (cloud[3] is None or cloud[2] == 1 or frame - cloud[3] >= cloud[2])
        ]
        if not changed and not due:
            return
        if any(points[path].ptr != previous.ptr for path, previous in self._geometry_points.items()):
            descriptors = []
            for path in self._geometry_points:
                descriptor = SceneDataFormat.Points()
                descriptor.points = points[path]
                descriptors.append(descriptor)
            self._geometry_sources.assign(descriptors)
            self._geometry_points = points.copy()
        max_count = max(len(values) for values in points.values())
        if changed and self._geometry_selection is not None:
            selection = self._geometry_selection
            selection.PrepareForReuse()
            wp.launch(
                _write_geometry,
                dim=(selection.GetCount(), max_count),
                inputs=[
                    wp.fabricarrayarray(data=selection, attrib="points", dtype=wp.vec3f),
                    wp.fabricarray(selection, "isaaclab:pointIndex"),
                    self._geometry_sources,
                ],
                device=self.device,
            )
        if self._host_geometry is not None and (due or changed and self._curve_selection is not None):
            targets, packed, host, host_sources = self._host_geometry
            wp.launch(
                _pack_geometry,
                dim=(len(points), max_count),
                inputs=[self._geometry_sources, targets],
                device=self.device,
            )
            wp.copy(host, packed)
            wp.synchronize_device(self.device)
            if changed and self._curve_selection is not None:
                selection = self._curve_selection
                selection.PrepareForReuse()
                wp.launch(
                    _write_geometry,
                    dim=(selection.GetCount(), max_count),
                    inputs=[
                        wp.fabricarrayarray(data=selection, attrib="points", dtype=wp.vec3f),
                        wp.fabricarray(selection, "isaaclab:pointIndex"),
                        host_sources,
                    ],
                    device="cpu",
                )
            with Sdf.ChangeBlock():
                for cloud in due:
                    attr, values, _, _, _ = cloud
                    attr.Set(Vt.Vec3fArray.FromNumpy(values.numpy()))
                    cloud[3:] = frame, version
        self._geometry_version = version

    def _bind_geometries(self, provider: SceneDataProvider, points: dict[str, wp.array]) -> None:
        """Bind exact published destinations; no scene discovery or physics-owned render metadata."""
        if not points:
            self._geometry_points = {}
            return
        # Foreign physics publishes world points. Author the sink once so Kit's USD refresh agrees.
        for path in points:
            geometry = UsdGeom.Xformable(provider.usd_stage.GetPrimAtPath(path))
            geometry.ClearXformOpOrder()
            geometry.SetResetXformStack(True)
        self.stage.SynchronizeToFabric()
        descriptors, host_ranges = [], {}
        host_count = 0
        mesh_count = curve_count = 0
        for index, (path, values) in enumerate(points.items()):
            descriptor = SceneDataFormat.Points()
            descriptor.points = values
            descriptors.append(descriptor)
            prim = self.stage.GetPrimAtPath(path)
            usdrt.Rt.Xformable(prim).SetWorldXformFromUsd()
            prim.CreateAttribute("isaaclab:pointIndex", usdrt.Sdf.ValueTypeNames.Int, custom=True).Set(index)
            prim_type = prim.GetTypeName()
            if prim_type == "Mesh":
                mesh_count += 1
                continue
            host_ranges[path] = (host_count, host_count + len(values))
            host_count += len(values)
            if prim_type == "BasisCurves":
                curve_count += 1
            elif prim_type == "Points":
                usd_points = UsdGeom.Points(provider.usd_stage.GetPrimAtPath(path))
                frequency = usd_points.GetPrim().GetAttribute("isaaclab:pointsUpdateFrequency").Get() or 1
                self._point_clouds.append([usd_points.GetPointsAttr(), path, frequency, None, -1])
            else:
                raise TypeError(f"Unsupported Fabric point destination {path}: {prim_type}.")
        self._geometry_sources = wp.array(descriptors, dtype=SceneDataFormat.Points, device=self.device)
        attrs = [
            (usdrt.Sdf.ValueTypeNames.Point3fArray, "points", usdrt.Usd.Access.ReadWrite),
            (usdrt.Sdf.ValueTypeNames.Int, "isaaclab:pointIndex", usdrt.Usd.Access.Read),
        ]
        if mesh_count:
            self._geometry_selection = self.stage.SelectPrims(
                require_attrs=attrs, require_prim_type="Mesh", device=self.device
            )
        if curve_count:
            # RTX Hydra ignores GPU Fabric BasisCurves points (NVBug 6502662).
            self._curve_selection = self.stage.SelectPrims(
                require_attrs=attrs, require_prim_type="BasisCurves", device="cpu"
            )
        if host_count:
            packed = wp.empty(host_count, dtype=wp.vec3f, device=self.device)
            host = wp.empty(host_count, dtype=wp.vec3f, device="cpu", pinned=wp.get_device(self.device).is_cuda)
            targets, host_sources = [], []
            for path in points:
                target, source = SceneDataFormat.Points(), SceneDataFormat.Points()
                if path in host_ranges:
                    start, end = host_ranges[path]
                    target.points, source.points = packed[start:end], host[start:end]
                targets.append(target)
                host_sources.append(source)
            self._host_geometry = (
                wp.array(targets, dtype=SceneDataFormat.Points, device=self.device),
                packed,
                host,
                wp.array(host_sources, dtype=SceneDataFormat.Points, device="cpu"),
            )
            for cloud in self._point_clouds:
                start, end = host_ranges[cloud[1]]
                cloud[1] = host[start:end]
        self._geometry_points = points.copy()

    def close(self) -> None:
        """Release stage-bound selections and borrowed SDP buffers."""
        self.transforms = self._selection = self._write_selection = None
        self._mapping = self._scales = self.hierarchy = self.stage = None
        self._geometry_points = self._geometry_sources = self._geometry_selection = self._curve_selection = None
        self._host_geometry = None
        self._point_clouds.clear()


@configclass
class FabricBackendCfg(BackendCfg):
    """Native Fabric identity; the stage is borrowed from the active simulation."""

    class_type: type = FabricBackend
    stage: Usd.Stage = field(kw_only=True, metadata={"copy": False})
    device: str = field(kw_only=True)
