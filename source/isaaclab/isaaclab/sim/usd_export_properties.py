# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Material ownership and mass-frame authoring for deployment USD."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pxr import Gf, Sdf, Usd, UsdPhysics, UsdShade

if TYPE_CHECKING:
    from isaaclab.assets import BaseArticulationData, BaseRigidObjectCollectionData, BaseRigidObjectData

    from .usd_export import UsdWriter


class UsdMassPropertiesWriter:
    """Author complete mass properties and body placement from public asset data."""

    def __init__(self, writer: UsdWriter):
        self.writer = writer

    def write_bodies(
        self,
        data: BaseArticulationData | BaseRigidObjectData | BaseRigidObjectCollectionData,
        paths: list[tuple[str, int]],
    ) -> None:
        """Write body placement and mass properties without copying transient velocities."""
        poses = data.body_link_pose_w.torch[self.writer.env_index].detach().cpu().numpy().reshape(-1, 7).copy()
        masses = data.body_mass.torch[self.writer.env_index].detach().cpu().numpy().reshape(-1)
        inertias = data.body_inertia.torch[self.writer.env_index].detach().cpu().numpy().reshape(-1, 3, 3)
        coms = data.body_com_pose_b.torch[self.writer.env_index].detach().cpu().numpy().reshape(-1, 7)
        if sorted(row for _, row in paths) != list(range(len(poses))):
            raise RuntimeError("Incomplete body identities for fixed configuration.")
        # Update parents first so child transforms retain their world placement.
        for path, row in sorted(paths, key=lambda item: Sdf.Path(item[0]).pathElementCount):
            prim = self.writer.stage.GetPrimAtPath(path)
            if path in self.writer.body_paths or not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f"Missing or multiply owned body {path}.")
            self.writer._write_body_pose(prim, poses[row])
            self.write(prim, masses[row], inertias[row], coms[row])
            self.writer.body_paths.add(path)

    def write(self, prim: Usd.Prim, mass: float, inertia: np.ndarray, com: np.ndarray) -> None:
        """Write mass [kg], COM pose [m, xyzw] and link-frame inertia [kg*m²]."""
        if not np.isfinite(mass) or mass < 0 or not np.isfinite(inertia).all() or not np.isfinite(com).all():
            raise ValueError(f"Invalid mass properties at {prim.GetPath()}.")
        if not np.allclose(inertia, inertia.T, atol=1e-7):
            raise ValueError(f"Non-symmetric inertia at {prim.GetPath()}.")
        rotation = Gf.Quatd(float(com[6]), Gf.Vec3d(*map(float, com[3:6])))
        axes = np.asarray(Gf.Matrix3d(rotation)).T
        # USD stores principal moments and axes, while public data supplies a full link-frame tensor.
        principal = axes.T @ inertia @ axes
        if np.allclose(principal, np.diag(np.diag(principal)), atol=1e-7):
            moments = np.diag(principal)
        else:
            moments, axes = np.linalg.eigh(inertia)
            # An eigenbasis may be reflected; quaternions require a right-handed frame.
            if np.linalg.det(axes) < 0:
                axes[:, 0] *= -1
            rotation = Gf.Matrix3d(*map(float, axes.T.flatten())).ExtractRotation().GetQuat()
        if np.any(moments < -1e-7):
            raise ValueError(f"Negative inertia at {prim.GetPath()}.")
        physics = UsdPhysics.MassAPI.Apply(prim)
        physics.CreateMassAttr().Set(float(mass))
        physics.CreateCenterOfMassAttr().Set(Gf.Vec3f(*map(float, com[:3])))
        physics.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*map(float, np.maximum(moments, 0))))
        physics.CreatePrincipalAxesAttr().Set(Gf.Quatf(rotation))


class UsdMaterialWriter:
    """Own independent material copies, connections and physics bindings."""

    def __init__(self, writer: UsdWriter):
        self.writer = writer

    def for_override(self, prim: Usd.Prim) -> UsdShade.Material:
        """Return an independently bound physics material, preserving its existing resources."""
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        original, _ = binding.ComputeBoundMaterial("physics")
        destination = prim.GetPath().AppendChild("ExportPhysicsMaterial")
        existing = self.writer.stage.GetPrimAtPath(destination)
        if existing and existing.GetCustomDataByKey("isaaclab:exportMaterial"):
            return UsdShade.Material(existing)
        if self.writer.stage.GetPrimAtPath(destination):
            raise RuntimeError(f"Export material path already exists: {destination}")
        if original:
            layer = self.writer.stage.GetRootLayer()
            if not Sdf.CopySpec(layer, original.GetPath(), layer, destination):
                raise RuntimeError(f"Cannot preserve bound material {original.GetPath()}.")
            material = UsdShade.Material(self.writer.stage.GetPrimAtPath(destination))
            self.writer._rebase_connections(material.GetPrim(), original.GetPath(), destination)
        else:
            material = UsdShade.Material.Define(self.writer.stage, destination)
        binding.Bind(material, bindingStrength=UsdShade.Tokens.weakerThanDescendants, materialPurpose="physics")
        effective, _ = binding.ComputeBoundMaterial("physics")
        if effective.GetPath() != destination:
            raise NotImplementedError(f"An ancestor material binding prevents contact overrides at {prim.GetPath()}.")
        material.GetPrim().SetCustomDataByKey("isaaclab:exportMaterial", True)
        return material

    def remove_unused_bindings(self) -> None:
        """Clear missing direct material targets only when resolved materials stay unchanged."""
        purposes = set(UsdShade.MaterialBindingAPI.GetMaterialPurposes()) | {"physics"}
        candidates = []
        for prim in self.writer.stage.Traverse():
            for relationship in prim.GetRelationships():
                tokens = relationship.GetName().split(":")
                if tokens[:2] != ["material", "binding"]:
                    continue
                if len(tokens) == 5 and tokens[2] == "collection":
                    purposes.add(tokens[3])
                elif len(tokens) in (2, 3):
                    purposes.add(tokens[2] if len(tokens) == 3 else "")
                    targets = relationship.GetTargets()
                    if (
                        len(targets) == 1
                        and targets[0].IsPrimPath()
                        and not self.writer.stage.GetPrimAtPath(targets[0])
                    ):
                        candidates.append(relationship)

        for relationship in candidates:
            bindings = [UsdShade.MaterialBindingAPI(prim) for prim in Usd.PrimRange(relationship.GetPrim())]

            def resolved_materials():
                return [
                    binding.ComputeBoundMaterial(purpose)[0].GetPath() for binding in bindings for purpose in purposes
                ]

            before = resolved_materials()
            targets = relationship.GetTargets()
            relationship.SetTargets([])
            # An invalid direct binding can mask an inherited material. Keep rejecting
            # that case instead of changing the appearance or physics-material fallback.
            if resolved_materials() != before:
                relationship.SetTargets(targets)
