# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publish one berry's deforming Gaussians to the RTX renderer on every frame, without leaving the GPU.

The berry's Gaussians are authored once in the render stage, from its asset. On every frame
:meth:`GaussianPublisher.deform` deforms and shades them from the tissue particles (:mod:`.binding`), and
:meth:`GaussianPublisher.publish` hands the renderer the resulting GPU arrays, which it reads in place. Only the
per-frame attributes change: positions, scales, orientations and the shading quaternion; opacities and spherical
harmonics stay as authored.
"""

import numpy as np
import warp as wp

from pxr import Sdf, Usd, UsdGeom, UsdShade, Vt

from .binding import GaussianBinding

SHADING = "primvars:squishyShQuaternion"
"""Per-Gaussian SH rotation and bruise tint read by the berry's MDL shader."""

# Per-frame Gaussian attributes and their number of float lanes.
_DYNAMIC = (("positions", 3), ("scales", 3), ("orientations", 4), (SHADING, 4))


@wp.kernel
def _bounds(x: wp.array[wp.vec3], s: wp.array[wp.vec3], bounds: wp.array2d[float]):
    i = wp.tid()
    reach = 3.0 * wp.max(s[i][0], wp.max(s[i][1], s[i][2]))
    for axis in range(3):
        wp.atomic_min(bounds, 0, axis, x[i][axis] - reach)
        wp.atomic_max(bounds, 1, axis, x[i][axis] + reach)


@wp.kernel
def _normalize(
    x: wp.array[wp.vec3],
    s: wp.array[wp.vec3],
    center: wp.vec3,
    rest_center: wp.vec3,
    factor: float,
    local_x: wp.array[wp.vec3],
    local_s: wp.array[wp.vec3],
):
    i = wp.tid()
    local_x[i] = (x[i] - center) / factor + rest_center
    local_s[i] = s[i] / factor


class GaussianPublisher:
    """Author one berry's Gaussians in the render stage and publish their deformation on every frame.

    Args:
        tissue: The berry's tissue (:class:`~..physics.tissue.BerryTissue`).
        root_path: Render-stage path under which the berry's Gaussians and material are authored.
    """

    def __init__(self, tissue, root_path: str):
        self.tissue = tissue
        self.root_path = root_path
        self.path = f"{root_path}/Gaussians/Field"
        self._bindings = {}
        asset = tissue.asset
        device = tissue.particles.device
        self.binding = GaussianBinding(asset, tissue.proxy["xyz"], tissue.proxy["regions"], device)
        # The renderer keeps the Gaussians' bounds from their authored extent. The deformed Gaussians are therefore
        # published in a local frame that keeps them inside their rest bounds, with a transform that moves them back
        # into place.
        self._rest_center = (asset["xyz"].min(0) + asset["xyz"].max(0)) / 2
        self._rest_half = (asset["xyz"].max(0) - asset["xyz"].min(0)) / 2
        self._transform = np.eye(4)
        count = len(asset["xyz"])
        with wp.ScopedDevice(device):
            self._extent = wp.empty((2, 3), dtype=float)
            # The renderer reads the published arrays in place, after the write returns. Two sets alternate: a set is
            # refilled only after the frame that read it has finished rendering.
            self._buffers = [
                {
                    "positions": wp.empty(count, dtype=wp.vec3),
                    "scales": wp.empty(count, dtype=wp.vec3),
                    "orientations": wp.empty(count, dtype=wp.quat),
                    SHADING: wp.empty(count, dtype=wp.vec4),
                }
                for _ in range(2)
            ]
        self._frame = 0

    def author_in_stage(self, stage: Usd.Stage) -> None:
        """Copy the berry's Gaussians and material from its asset into the render stage."""
        source = Usd.Stage.Open(self.tissue.usd_stage.GetRootLayer().identifier)
        layer = source.Flatten()
        Sdf.CreatePrimInLayer(stage.GetRootLayer(), self.root_path)
        Sdf.CopySpec(layer, "/Berry", stage.GetRootLayer(), self.root_path)
        # The physics data is not part of the render scene.
        stage.RemovePrim(f"{self.root_path}/TaskData")
        stage.RemovePrim(f"{self.root_path}/Tissue")
        # The renderer only loads a Gaussian field copied on its own under a defined transform, not one copied as
        # part of the asset's tree.
        stage.RemovePrim(f"{self.root_path}/Gaussians")
        UsdGeom.Xform.Define(stage, f"{self.root_path}/Gaussians")
        Sdf.CopySpec(layer, "/Berry/Gaussians", stage.GetRootLayer(), self.path)
        prim = stage.GetPrimAtPath(self.path)
        for attr in prim.GetAttributes():
            if attr.GetName().startswith("berry:"):
                prim.RemoveProperty(attr.GetName())
        # The berry may have been turned and moved into the punnet: author its placed Gaussians. Time samples mark the
        # attributes the renderer should expect to change.
        original = source.GetPrimAtPath("/Berry/Gaussians")
        for name, key, array_type in (
            ("positions", "xyz", Vt.Vec3fArray),
            ("scales", "scales", Vt.Vec3fArray),
            ("orientations", "rotations", Vt.QuatfArray),
            ("opacities", "alpha", Vt.FloatArray),
            ("radiance:sphericalHarmonicsCoefficients", "sh", Vt.Vec3fArray),
        ):
            array = self.tissue.asset[key].reshape(-1, 3) if key == "sh" else self.tissue.asset[key]
            value = array_type.FromNumpy(np.ascontiguousarray(array, np.float32))
            attr = prim.GetAttribute(name)
            attr.Clear()
            attr.Set(value)
            if original.GetAttribute(name).GetNumTimeSamples():
                attr.Set(value, 0)
                attr.Set(value, 1)
        shading = prim.GetAttribute(SHADING)
        value = shading.Get()
        shading.Clear()
        for time in (Usd.TimeCode.Default(), 0, 1):
            shading.Set(value, time)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            UsdShade.Material.Get(stage, f"{self.root_path}/Materials/Radiance")
        )

    def bind_to_renderer(self, rtx) -> None:
        """Bind the per-frame attributes to the running renderer."""
        from ovrtx import BindingFlag

        # Persistent bindings establish the renderer's animated-geometry dataflow; by-name writes can leave its
        # acceleration structure stale.
        self._rtx = rtx
        self._bindings = {
            name: rtx.bind_array_attribute(
                [self.path], name, dtype="float32", shape=(lanes,), flags=BindingFlag.OPTIMIZE
            )
            for name, lanes in _DYNAMIC
        }

    def deform(self) -> tuple[np.ndarray, dict]:
        """Deform and shade the Gaussians on the GPU; return their transform and the arrays to publish."""
        binding = self.binding
        with wp.ScopedDevice(self.tissue.particles.device):
            binding.deform(self.tissue.positions_warp(), self.tissue.damage)
            out = self._buffers[self._frame % 2]
            self._frame += 1
            # Local frame: center the deformed Gaussians on their rest center and shrink them into their rest bounds.
            self._extent.assign(np.array([[np.inf] * 3, [-np.inf] * 3], np.float32))
            wp.launch(_bounds, dim=len(binding.positions), inputs=[binding.positions, binding.scales, self._extent])
            extent = self._extent.numpy().astype(np.float64)
            center = extent.mean(0)
            factor = float(np.max((extent[1] - extent[0]) / 2 / self._rest_half))
            if not np.isfinite(factor) or factor <= 0:
                raise ValueError("Invalid deformed Gaussian bounds")
            wp.launch(
                _normalize,
                dim=len(binding.positions),
                inputs=[binding.positions, binding.scales, wp.vec3(*center), wp.vec3(*self._rest_center), factor],
                outputs=[out["positions"], out["scales"]],
            )
            wp.copy(out["orientations"], binding.orientations)
            wp.copy(out[SHADING], binding.shading)
        # Row-vector convention: scale, then translate back to the deformed center and to the berry's place.
        transform = np.eye(4)
        transform[:3, :3] *= factor
        transform[3, :3] = center - factor * self._rest_center + self.tissue.offset
        return transform, out

    def publish(self, prepared: tuple[np.ndarray, dict]) -> None:
        """Publish arrays from :meth:`deform` to the renderer."""
        from ovrtx import DataAccess, Semantic

        transform, values = prepared
        self._rtx.write_attribute(
            prim_paths=[self.path], attribute_name="omni:xform", tensor=transform[None], semantic=Semantic.XFORM_MAT4x4
        )
        cuda_stream = wp.get_stream(self.tissue.particles.device).cuda_stream
        for name, array in values.items():
            # The renderer reads the GPU buffers in place, ordered after the kernels on their stream.
            self._bindings[name].write([array], data_access=DataAccess.ASYNC, cuda_stream=cuda_stream)

    def close(self) -> None:
        for binding in self._bindings.values():
            binding.unbind()
