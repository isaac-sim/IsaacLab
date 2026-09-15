# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX source identities for fixed scene export in the PhysX target dialect."""

from pxr import Usd, UsdPhysics

from isaaclab.sim.usd_export import AssetPaths


class SceneAdapter:
    """Resolve body/joint names only among matching physical schemas."""

    def __init__(self, scene):
        self.scene = scene

    def paths(self, asset) -> AssetPaths:
        if asset not in self.scene.articulations.values():
            return AssetPaths([(str(path), row) for row, path in enumerate(asset.root_view.prim_paths)], [])
        roots = asset.root_view.prim_paths
        if len(roots) != 1:
            raise NotImplementedError("Register each articulation instance separately for fixed export.")
        root = asset.stage.GetPrimAtPath(roots[0])
        # An articulation-root API may be on a link; find the nearest scope containing every DOF/link.
        while root and not root.IsPseudoRoot():
            bodies, joints = {}, {}
            for prim in Usd.PrimRange(root):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    bodies.setdefault(prim.GetName(), []).append(str(prim.GetPath()))
                elif prim.IsA(UsdPhysics.Joint):
                    joints.setdefault(prim.GetName(), []).append(str(prim.GetPath()))
            if set(asset.body_names) <= bodies.keys() and set(asset.joint_names) <= joints.keys():
                break
            root = root.GetParent()

        def resolve(names, paths):
            result = []
            for row, name in enumerate(names):
                matches = paths.get(name, [])
                if len(matches) != 1:
                    raise RuntimeError(f"Ambiguous or missing physical identity {name}: {matches}")
                result.append((matches[0], row))
            return result

        return AssetPaths(resolve(asset.body_names, bodies), resolve(asset.joint_names, joints))

    def write_extensions(self, writer) -> None:
        """Fixed OVPhysX schemas are already authored; common mappings cover initialized joints."""
