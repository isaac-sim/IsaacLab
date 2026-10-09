# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render-only NVIDIA kitchen composition around the coffee worktable."""

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

KITCHEN_USD = f"{ISAAC_NUCLEUS_DIR}/Environments/replicator_kitchen/kitchen_l_island.usda"


def add_kitchen(stage: Usd.Stage, parent: str) -> None:
    """Reference the public room after physics construction, with physics disabled.

    The original worktable occupies the sample kitchen's island position. Strong
    inactive opinions exclude the island and its props before their references
    compose, avoiding overlapping counters and unnecessary asset downloads.
    """
    kitchen = UsdGeom.Xform.Define(stage, parent)
    excluded = (
        "IslandBase_00",
        "IslandBase_01",
        "IslandBase_02",
        "CapSurface_03",
        "Island_SkirtFront",
        "Island_SkirtBack",
        "Island_SkirtLeft",
        "Island_SkirtRight",
        "BowlF01",
        "BagelC01",
        "Toaster",
        "BananaA01_01",
        "CuttingBoardB01",
        "AppleA01",
        "CoffeeMugA01",
        "CoffeeCupB01",
        "PhysicsScene",
    )
    for name in excluded:
        stage.OverridePrim(f"{parent}/{name}").SetActive(False)
    kitchen.GetPrim().GetReferences().AddReference(KITCHEN_USD)
    if not stage.GetPrimAtPath(f"{parent}/KitchenRoom/Floor"):
        raise RuntimeError(f"The NVIDIA kitchen did not resolve: {KITCHEN_USD}")
    kitchen.AddTranslateOp().Set(Gf.Vec3d(0.6351131432, -0.5645103994, -0.746))
    for prim in Usd.PrimRange(kitchen.GetPrim()):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if prim.IsA(UsdLux.DomeLight):
            UsdLux.DomeLight(prim).CreateIntensityAttr(1800.0)
        elif prim.HasAPI(UsdLux.LightAPI):
            light = UsdLux.LightAPI(prim)
            light.CreateEnableColorTemperatureAttr(True)
            light.CreateColorTemperatureAttr(3200.0)
            light.CreateIntensityAttr(500.0 if "UnderCabinet" in str(prim.GetPath()) else 1400.0)
    fill = UsdLux.RectLight.Define(stage, f"{parent}/RoomFill")
    fill.CreateWidthAttr(2.0)
    fill.CreateHeightAttr(1.4)
    fill.CreateIntensityAttr(1400.0)
    fill.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.87))
    transform = Gf.Matrix4d().SetLookAt(Gf.Vec3d(1.5, -1.4, 1.5), Gf.Vec3d(0.2, 1.7, 0.5), Gf.Vec3d(0, 0, 1))
    # This emitter is authored in world coordinates, independently of the sample's placement.
    xform = UsdGeom.Xformable(fill)
    xform.SetResetXformStack(True)
    xform.AddTransformOp().Set(transform.GetInverse())
    # Matte painted cabinetry, warm plaster and quiet brass avoid busy reflections.
    for name, color, roughness, metallic in (
        ("wall", (0.68, 0.64, 0.55), 0.85, 0.0),
        ("ceiling", (0.75, 0.72, 0.64), 0.90, 0.0),
        ("carcass", (0.13, 0.22, 0.17), 0.55, 0.0),
        ("handle", (0.44, 0.28, 0.11), 0.35, 0.75),
        ("pendant", (0.38, 0.24, 0.10), 0.40, 0.70),
    ):
        material = UsdShade.Material.Get(stage, f"{parent}/Looks/KitchenFinish_{name}")
        if not material:
            continue
        shader = UsdShade.Shader.Define(stage, f"{material.GetPath()}/CoffeeFinish")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        material.CreateSurfaceOutput("mdl").DisconnectSource()
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
