# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-berry persistent Gaussian geometry and deformation/shading bindings."""

from types import SimpleNamespace

import numpy as np
import warp as wp

from pxr import Sdf, Usd, UsdGeom, UsdShade, Vt

from ..physics.mpm.binding import make_binding
from ..physics.mpm.rtx_gaussian_frame import GaussianLocalFrame
from ..physics.mpm.rtx_sh_frame import SHMaterialFrame


class BerryGaussianStream:
    """Publish one independent berry without sharing its material or GPU state."""

    def __init__(self, berry, root_path, partitions, hide_interior):
        self.berry = berry
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
        with wp.ScopedDevice(self.berry.device):
            self.binding.deform_gpu(self.berry.sim.x, host=False)
            self.frame = GaussianLocalFrame(self.berry.asset["xyz"], self.berry.asset["scales"], self.berry.device)
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
            for name, lanes in (
                ("positions", 3),
                ("scales", 3),
                ("orientations", 4),
                ("primvars:squishyShQuaternion", 4),
            )
        }

    def prepare(self, sh_rotation):
        with wp.ScopedDevice(self.berry.device):
            xyz, scales, quats = self.binding.deform_gpu(self.berry.sim.x, host=False)
            xyz, scales, transform = self.frame.evaluate(xyz, scales)
            shading = self.sh_frame.evaluate(self.berry.sim.damage, rotate=sh_rotation)
            values = {
                "positions": xyz.numpy(),
                "scales": scales.numpy(),
                "orientations": quats.numpy(),
                **{name: array.numpy() for name, array in shading.items()},
            }
        transform = transform.copy()
        transform[3, :3] += self.berry.offset
        return transform, values

    def update(self, prepared):
        from ovrtx import Semantic

        transform, values = prepared
        self._rtx.write_attribute(
            prim_paths=self.paths,
            attribute_name="omni:xform",
            tensor=np.repeat(transform[None], len(self.paths), axis=0),
            semantic=Semantic.XFORM_MAT4x4,
        )
        for name, value in values.items():
            self._berry_bindings[name].write([np.ascontiguousarray(value[idx]) for idx in self.indices])

    def verify(self, prepared):
        """Read back the native renderer's complete published Gaussian arrays."""
        _, values = prepared
        for name, expected in values.items():
            restored = self._rtx.read_array_attribute(attribute_name=name, prim_paths=self.paths)
            for path, idx in zip(self.paths, self.indices):
                np.testing.assert_allclose(
                    np.from_dlpack(restored[path]).reshape(expected[idx].shape),
                    expected[idx],
                    atol=1e-7,
                )
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
