# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""EBC room placement around the freestanding berry demo table."""

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from isaaclab.utils.assets import retrieve_file_path

from ..assets.asset_root import background_root

# Measured in background/ebc/aligned.usda (point_cloud_aligned_new alignment).
# Leave an approximately one-metre aisle between the table and coffee counter.
# Robot base and table top stay at task z=0; do not rotate the IK/MPM frames.
EBC_BASE_POSITION = (-0.8, -1.8, 0.0)
EBC_COUNTER_YAW_DEG = 86.0


def ebc_background_path() -> str:
    """Return the source path (Nucleus URL or local override) for the EBC alignment layer."""
    return f"{background_root()}/aligned.usda"


def ebc_to_task_transform() -> Gf.Matrix4d:
    """Map aligned EBC coordinates [m] into the freestanding workcell frame."""
    robot_to_ebc = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), EBC_COUNTER_YAW_DEG))
    robot_to_ebc.SetTranslateOnly(Gf.Vec3d(*EBC_BASE_POSITION))
    return robot_to_ebc.GetInverse()


def add_ebc_background(stage: Usd.Stage, radiance_shader: UsdShade.Shader) -> str:
    """Reference the static room and bind the packaged SH shader (no physics import).

    Avoid the legacy scan's unqualified ParticleFieldEmissive.mdl dependency.
    Reuse the berry USDZ's resolved shader, with its identity material-frame
    fallback and no per-berry bruise attributes.
    """
    path = retrieve_file_path(ebc_background_path())
    source = Usd.Stage.Open(path)
    original = source.GetPrimAtPath("/World/GaussianBackground/ParticleField")
    if not original or original.GetTypeName() != "ParticleField3DGaussianSplat":
        raise ValueError(f"Expected EBC ParticleField in {path}")
    # The corrected alignment targets the raw Gaussian arrays, as imported by
    # the Gaussian-twin task. Drop the legacy field-local Y/Z flip and raster
    # hints, and author one affine transform for native RTX. Explicitly use the
    # parent: XformCache ignores unknown ParticleField types in kitless USD but
    # would include the legacy flip if its schema were registered.
    # Keep the large arrays referenced, not duplicated in a temporary stage.
    UsdGeom.Xform.Define(stage, "/World/EBC")
    field = stage.DefinePrim("/World/EBC/Gaussians", "ParticleField3DGaussianSplat")
    field.GetReferences().AddReference(path, original.GetPath())
    transform = UsdGeom.Xformable(field)
    transform.ClearXformOpOrder()
    transform.AddTransformOp().Set(
        UsdGeom.XformCache().GetLocalToWorldTransform(original.GetParent()) * ebc_to_task_transform()
    )
    for name in ("projectionModeHint", "sortingModeHint"):
        if field.HasAttribute(name):
            field.GetAttribute(name).Block()
    material = UsdShade.Material.Define(stage, "/World/EBCMaterial")
    shader = UsdShade.Shader.Define(stage, "/World/EBCMaterial/Shader")
    shader.SetSourceAsset(radiance_shader.GetSourceAsset("mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier(radiance_shader.GetSourceAssetSubIdentifier("mdl"), "mdl")
    shader.CreateInput("emission_intensity", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    UsdShade.MaterialBindingAPI.Apply(field).Bind(material)
    return path
