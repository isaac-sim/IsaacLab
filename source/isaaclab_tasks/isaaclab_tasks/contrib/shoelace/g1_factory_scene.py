# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visual packing workstation for the G1 shoelace task."""

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils

# Authored tabletop height with the workstation at its native scale [m].
PACKING_TABLE_HEIGHT = 0.9940513388652052


@sim_utils.clone
def spawn_packing_workstation(
    prim_path: str,
    cfg: sim_utils.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: object,
) -> Usd.Prim:
    """Place a visual-only packing table and move its tray clear of the manipulation area.

    The shoe is kinematic and needs no supporting pedestal. Removing imported physics
    before cloning keeps the decorative table and crates out of task contacts.
    The caller must make instances editable so nested crate colliders are included.
    """
    if not cfg.make_uninstanceable:
        raise ValueError("The packing workstation requires make_uninstanceable=True.")
    root = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    tray = root.GetStage().GetPrimAtPath(root.GetPath().AppendChild("container_h20"))
    if not tray.IsValid():
        raise ValueError("PackingTable asset is missing its container_h20 tray.")
    # World tray center (0.70, 0.23), relative to the table root at (0, 0.10).
    tray.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.70, 0.13, PACKING_TABLE_HEIGHT))
    tray.GetAttribute("xformOp:scale").Set(Gf.Vec3f(0.9, 0.9, 0.75))
    # The source uses MDL materials; explicit USD preview surfaces also render in Newton GL.
    materials = {}
    for name, color in {
        "TableTop": (0.42, 0.46, 0.48),
        "TableTopSide": (0.30, 0.23, 0.15),
        "Body": (0.12, 0.16, 0.20),
        "Bolts": (0.30, 0.34, 0.38),
        "BlueCrate": (0.035, 0.18, 0.29),
        "YellowCrate": (0.55, 0.36, 0.08),
        "Tray": (0.045, 0.22, 0.23),
    }.items():
        material_path = str(root.GetPath().AppendPath(f"PreviewLooks/{name}"))
        sim_utils.spawn_preview_surface(
            material_path,
            sim_utils.PreviewSurfaceCfg(
                diffuse_color=color, roughness=0.65, metallic=0.5 if name in ("Body", "Bolts") else 0.0
            ),
        )
        materials[name] = UsdShade.Material.Get(root.GetStage(), material_path)
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh):
            path = str(prim.GetPath())
            material_name = prim.GetName().removeprefix("M_HeavyDutyPackingTable_C01_")
            if "/SM_Crate_" in path:
                material_name = "BlueCrate" if "Blue" in path else "YellowCrate"
            elif "/container_h20/" in path:
                material_name = "Tray"
            if material_name in materials:
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    materials[material_name], bindingStrength=UsdShade.Tokens.strongerThanDescendants
                )
        # Exclude decorative rigid bodies and colliders from Newton import, not just contact pairs.
        for api in (UsdPhysics.CollisionAPI, UsdPhysics.MeshCollisionAPI, UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI):
            if prim.HasAPI(api):
                prim.RemoveAPI(api)
        if prim.IsA(UsdPhysics.Joint):
            UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
    return root
