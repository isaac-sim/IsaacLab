# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The scanned room around the workcell: a static Gaussian scan of an office coffee corner (the EBC scan)."""

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from isaaclab.utils.assets import retrieve_file_path

from ..assets.asset_paths import room_scan_root

# Robot base pose in the scan of background/ebc/aligned.usda (point_cloud_aligned_new alignment).
# Leave an approximately one-metre aisle between the table and coffee counter.
# Robot base and table top stay at task z=0; do not rotate the IK/MPM frames.
ROBOT_BASE_IN_ROOM = (-0.8, -1.8, 0.0)
ROBOT_YAW_IN_ROOM_DEG = 86.0


def room_scan_path() -> str:
    """Return the source path (Nucleus URL or local override) of the room scan's alignment layer."""
    return f"{room_scan_root()}/aligned.usda"


def room_to_task_transform() -> Gf.Matrix4d:
    """Map the room scan's coordinates [m] into the workcell frame."""
    robot_in_room = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), ROBOT_YAW_IN_ROOM_DEG))
    robot_in_room.SetTranslateOnly(Gf.Vec3d(*ROBOT_BASE_IN_ROOM))
    return robot_in_room.GetInverse()


def add_room_scan(stage: Usd.Stage, radiance_shader: UsdShade.Shader) -> str:
    """Reference the static room and bind the packaged SH shader (no physics import).

    Avoid the legacy scan's unqualified ParticleFieldEmissive.mdl dependency.
    Reuse the berry USDZ's resolved shader, with its identity material-frame
    fallback and no per-berry bruise attributes.
    """
    path = retrieve_file_path(room_scan_path())
    source = Usd.Stage.Open(path)
    original = source.GetPrimAtPath("/World/GaussianBackground/ParticleField")
    if not original or original.GetTypeName() != "ParticleField3DGaussianSplat":
        raise ValueError(f"Expected a Gaussian ParticleField in {path}")
    # The corrected alignment targets the raw Gaussian arrays, as imported by
    # the Gaussian-twin task. Drop the legacy field-local Y/Z flip and raster
    # hints, and author one affine transform for native RTX. Explicitly use the
    # parent: XformCache ignores unknown ParticleField types in kitless USD but
    # would include the legacy flip if its schema were registered.
    # Keep the large arrays referenced, not duplicated in a temporary stage.
    UsdGeom.Xform.Define(stage, "/World/Room")
    field = stage.DefinePrim("/World/Room/Gaussians", "ParticleField3DGaussianSplat")
    field.GetReferences().AddReference(path, original.GetPath())
    transform = UsdGeom.Xformable(field)
    transform.ClearXformOpOrder()
    transform.AddTransformOp().Set(
        UsdGeom.XformCache().GetLocalToWorldTransform(original.GetParent()) * room_to_task_transform()
    )
    for name in ("projectionModeHint", "sortingModeHint"):
        if field.HasAttribute(name):
            field.GetAttribute(name).Block()
    material = UsdShade.Material.Define(stage, "/World/RoomMaterial")
    shader = UsdShade.Shader.Define(stage, "/World/RoomMaterial/Shader")
    shader.SetSourceAsset(radiance_shader.GetSourceAsset("mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier(radiance_shader.GetSourceAssetSubIdentifier("mdl"), "mdl")
    shader.CreateInput("emission_intensity", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    UsdShade.MaterialBindingAPI.Apply(field).Bind(material)
    return path
