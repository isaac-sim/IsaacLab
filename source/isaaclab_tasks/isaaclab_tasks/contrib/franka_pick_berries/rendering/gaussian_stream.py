# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-berry persistent Gaussian geometry and deformation/shading bindings."""

from types import SimpleNamespace

import numpy as np
import warp as wp

from pxr import Sdf, Usd, UsdGeom, UsdShade, Vt

from .binding import make_binding
from .gaussian_frame import GaussianLocalFrame
from .sh_frame import SHMaterialFrame


@wp.kernel
def gather_vec3(source: wp.array[wp.vec3], indices: wp.array[int], target: wp.array[wp.vec3]):
    i = wp.tid()
    target[i] = source[indices[i]]


@wp.kernel
def gather_vec4(source: wp.array[wp.vec4], indices: wp.array[int], target: wp.array[wp.vec4]):
    i = wp.tid()
    target[i] = source[indices[i]]


@wp.kernel
def gather_quat(source: wp.array[wp.quat], indices: wp.array[int], target: wp.array[wp.quat]):
    i = wp.tid()
    target[i] = source[indices[i]]


_GATHER = {wp.vec3: gather_vec3, wp.vec4: gather_vec4, wp.quat: gather_quat}


# Per-frame Gaussian attributes and their number of float lanes: the deformed geometry and the shading frame.
_DYNAMIC = (("positions", 3), ("scales", 3), ("orientations", 4), ("primvars:squishyShQuaternion", 4))


class BerryGaussianStream:
    """Publish one independent berry without sharing its material or GPU state."""

    def __init__(self, berry, root_path, partitions, hide_interior):
        self.berry = berry
        self._berry_bindings = {}
        self.root_path = root_path
        self.path = f"{root_path}/Gaussians"
        visible = np.arange(len(self.berry.asset["xyz"]))
        if hide_interior:
            prim = self.berry.usd_stage.GetPrimAtPath("/Berry/Gaussians")
            interior = np.asarray(prim.GetAttribute("primvars:interior").Get())
            if interior.shape != visible.shape:
                raise ValueError("Expected one interior flag per Gaussian")
            visible = visible[interior == 0]
        if len(visible) < partitions:
            raise ValueError("Expected at least one visible Gaussian per partition")
        self.indices = [visible[i::partitions] for i in range(partitions)]
        self.paths = [f"{self.path}/Part{i}" for i in range(partitions)]
        self.binding = make_binding(
            self.berry.asset,
            self.berry.proxy["xyz"],
            self.berry.proxy["regions"],
            self.berry.profile["simulation"]["binding"],
        )
        device = self.berry.particles.device
        with wp.ScopedDevice(device):
            self.binding.deform_gpu(self.berry.positions_wp(), host=False)
            self.frame = GaussianLocalFrame(self.berry.asset["xyz"], self.berry.asset["scales"], device)
            lower = np.max([self.berry.asset["xyz"][idx].min(0) for idx in self.indices], axis=0)
            upper = np.min([self.berry.asset["xyz"][idx].max(0) for idx in self.indices], axis=0)
            self.frame.center, self.frame.half = (
                (lower + upper) / 2,
                (upper - lower) / 2,
            )
            if np.any(self.frame.half <= 0):
                raise ValueError("Partition bounds have no common volume")
            sh = SimpleNamespace(
                binding=self.binding,
                inverse=wp.array(np.linalg.inv(self.binding.base).astype(np.float32), dtype=wp.mat33),
                nearest=wp.array(self.binding.ids[:, 0].copy(), dtype=int),
            )
            self.sh_frame = SHMaterialFrame(sh, quaternion=True)
            # Each partition's Gaussians are gathered on the GPU into buffers that the renderer reads without a
            # copy, after the write returns. Two sets alternate: a set is refilled only after the frame that read
            # it has finished rendering. The buffers take their source's element type on the first frame.
            self._indices = [wp.array(idx.astype(np.int32), dtype=int) for idx in self.indices]
            self._buffers = ({}, {})
            self._frame = 0

    def author(self, stage):
        self.stage = stage
        source = Usd.Stage.Open(self.berry.usd_stage.GetRootLayer().identifier)
        layer = source.Flatten()
        Sdf.CreatePrimInLayer(self.stage.GetRootLayer(), self.root_path)
        Sdf.CopySpec(layer, "/Berry", self.stage.GetRootLayer(), self.root_path)
        # The physics/authoring metadata is not part of the render scene.
        self.stage.RemovePrim(f"{self.root_path}/TaskData")
        self.stage.RemovePrim(f"{self.root_path}/Tissue")
        self.stage.RemovePrim(self.path)
        UsdGeom.Xform.Define(self.stage, self.path)
        original = source.GetPrimAtPath("/Berry/Gaussians")
        for path, idx in zip(self.paths, self.indices):
            Sdf.CopySpec(layer, "/Berry/Gaussians", self.stage.GetRootLayer(), path)
            prim = self.stage.GetPrimAtPath(path)
            for attr in prim.GetAttributes():
                if attr.GetName().startswith("berry:"):
                    prim.RemoveProperty(attr.GetName())
            for name, key, ctor in (
                ("positions", "xyz", Vt.Vec3fArray),
                ("scales", "scales", Vt.Vec3fArray),
                ("orientations", "rotations", Vt.QuatfArray),
                ("opacities", "alpha", Vt.FloatArray),
                ("radiance:sphericalHarmonicsCoefficients", "sh", Vt.Vec3fArray),
            ):
                array = self.berry.asset[key][idx]
                if key == "sh":
                    array = array.reshape(-1, 3)
                value = ctor.FromNumpy(np.ascontiguousarray(array, np.float32))
                attr = prim.GetAttribute(name)
                attr.Clear()
                attr.Set(value)
                if original.GetAttribute(name).GetNumTimeSamples():
                    attr.Set(value, 0)
                    attr.Set(value, 1)
            for name, ctor in (
                ("primvars:squishyShQuaternion", Vt.Vec4fArray),
                ("primvars:interior", Vt.IntArray),
                ("primvars:repairedSkin", Vt.IntArray),
            ):
                if not original.HasAttribute(name):
                    continue
                values = np.asarray(original.GetAttribute(name).Get())[idx]
                attr = prim.GetAttribute(name)
                attr.Clear()
                attr.Set(ctor.FromNumpy(np.ascontiguousarray(values)))
                if name.endswith("Quaternion"):
                    attr.Set(attr.Get(), 0)
                    attr.Set(attr.Get(), 1)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                UsdShade.Material.Get(self.stage, f"{self.root_path}/Materials/Radiance")
            )

    def bind(self, rtx):
        self._rtx = rtx
        from ovrtx import BindingFlag

        # Persistent bindings establish OVRTX's animated-geometry dataflow.
        # By-name writes can read back correctly while leaving its BVH stale.
        self._berry_bindings = {
            name: self._rtx.bind_array_attribute(
                self.paths, name, dtype="float32", shape=(lanes,), flags=BindingFlag.OPTIMIZE
            )
            for name, lanes in _DYNAMIC
        }

    def prepare(self, sh_rotation):
        """Deform and shade the Gaussians and gather each partition's arrays, all on the GPU."""
        with wp.ScopedDevice(self.berry.particles.device):
            xyz, scales, quats = self.binding.deform_gpu(self.berry.positions_wp(), host=False)
            xyz, scales, transform = self.frame.evaluate(xyz, scales)
            shading = self.sh_frame.evaluate(self.berry.damage, rotate=sh_rotation)
            sources = {"positions": xyz, "scales": scales, "orientations": quats, **shading}
            buffers = self._buffers[self._frame % 2]
            self._frame += 1
            for name, _ in _DYNAMIC:
                source = sources[name]
                if name not in buffers:
                    buffers[name] = [wp.empty(len(idx), dtype=source.dtype) for idx in self._indices]
                for indices, part in zip(self._indices, buffers[name]):
                    wp.launch(_GATHER[source.dtype], dim=len(part), inputs=[source, indices, part])
        transform = transform.copy()
        transform[3, :3] += self.berry.offset
        return transform, buffers

    def update(self, prepared):
        from ovrtx import Semantic

        transform, values = prepared
        self._rtx.write_attribute(
            prim_paths=self.paths,
            attribute_name="omni:xform",
            tensor=np.repeat(transform[None], len(self.paths), axis=0),
            semantic=Semantic.XFORM_MAT4x4,
        )
        from ovrtx import DataAccess

        stream = wp.get_stream(self.berry.particles.device).cuda_stream
        for name, parts in values.items():
            # The renderer reads the GPU buffers in place, ordered after the gathers on their stream.
            self._berry_bindings[name].write(parts, data_access=DataAccess.ASYNC, cuda_stream=stream)

    def verify(self, prepared):
        """Read back the native renderer's complete published Gaussian arrays."""
        _, values = prepared
        for name, parts in values.items():
            restored = self._rtx.read_array_attribute(attribute_name=name, prim_paths=self.paths)
            for path, part in zip(self.paths, parts):
                expected = part.numpy()
                np.testing.assert_allclose(np.from_dlpack(restored[path]).reshape(expected.shape), expected, atol=1e-7)
        for name, key in (
            ("opacities", "alpha"),
            ("radiance:sphericalHarmonicsCoefficients", "sh"),
        ):
            restored = self._rtx.read_array_attribute(attribute_name=name, prim_paths=self.paths)
            for path, idx in zip(self.paths, self.indices):
                expected = self.berry.asset[key][idx]
                np.testing.assert_array_equal(np.from_dlpack(restored[path]).reshape(expected.shape), expected)
        return True

    def close(self):
        for binding in self._berry_bindings.values():
            binding.unbind()
