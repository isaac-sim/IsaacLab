# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RTX spectator view matching the recorded camera poses."""

import math

import numpy as np


def prepare_view(env, visualizer):
    """Set the fixed camera and separate visual surfaces after physics import."""
    import omni.kit.app  # noqa: PLC0415
    from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

    from isaaclab.sim import get_current_stage  # noqa: PLC0415

    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.replicator.core", True)
    import omni.replicator.core as rep  # noqa: PLC0415

    rep.settings.set_render_rtx_realtime(antialiasing="DLAA")
    stage = get_current_stage()
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/Robot")):
        if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.GetName().startswith("imu_in_"):
            UsdGeom.Imageable(prim).MakeInvisible()
    # Kit's viewport camera opinions in the session layer override the scene layer.
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        camera_path = "/OmniverseKit_Persp"
        camera = UsdGeom.Camera.Define(stage, camera_path)
        transform = UsdGeom.Xformable(camera.GetPrim())
        transform.ClearXformOpOrder()
        transform.SetResetXformStack(True)
        view = Gf.Matrix4d(1.0).SetLookAt(
            Gf.Vec3d(*env.cfg.camera_eye), Gf.Vec3d(*env.cfg.camera_target), Gf.Vec3d(0.0, 0.0, 1.0)
        )
        transform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(view.GetInverse())
        aperture = 20.955 * visualizer.cfg.window.size[1] / visualizer.cfg.window.size[0]
        camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateVerticalApertureAttr(aperture)
        camera.CreateFocalLengthAttr(aperture / (2 * math.tan(math.radians(env.cfg.camera_vfov) / 2)))
    for _ in range(60):
        env.sim.render()
        visualizer.render_rgb_array()
    env.sim.forward()
    env.scene.update(0.0)
    env.scene["depth_camera"].update(0.0, force_recompute=True)
    _separate_ramp_base(stage)
    for _ in range(30):
        env.sim.render()
        visualizer.render_rgb_array()


def _separate_ramp_base(stage):
    """Inset the visible base by 1 mm while preserving collision and depth geometry."""
    from pxr import Sdf, UsdGeom, UsdShade, Vt  # noqa: PLC0415

    base = next(prim for prim in stage.Traverse() if prim.GetName() == "wood_base")
    wood = next(prim for prim in stage.Traverse() if prim.GetName() == "wood_0_0")
    source = UsdGeom.Mesh(base)
    points = np.asarray(source.GetPointsAttr().Get(), dtype=np.float32).copy()
    lower, upper = points.min(axis=0), points.max(axis=0)
    center = 0.5 * (lower + upper)
    points[:, :2] = center[:2] + (points[:, :2] - center[:2]) * (1.0 - 0.002 / (upper - lower)[:2])
    visual = UsdGeom.Mesh.Define(stage, base.GetPath().GetParentPath().AppendChild("wood_base_visual"))
    visual.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    visual.CreateFaceVertexCountsAttr(source.GetFaceVertexCountsAttr().Get())
    visual.CreateFaceVertexIndicesAttr(source.GetFaceVertexIndicesAttr().Get())
    visual.CreateSubdivisionSchemeAttr("none")
    UsdGeom.PrimvarsAPI(visual).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(
        UsdGeom.PrimvarsAPI(source).GetPrimvar("st").Get()
    )
    UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(
        UsdShade.MaterialBindingAPI(base).ComputeBoundMaterial()[0]
    )
    faces = np.asarray(source.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    top_faces = np.flatnonzero(np.all(np.isclose(points[faces, 2], upper[2]), axis=1))
    subset = UsdGeom.Subset.CreateGeomSubset(
        visual, "wood_top", UsdGeom.Tokens.face, top_faces.tolist(), "materialBind"
    )
    UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(
        UsdShade.MaterialBindingAPI(wood).ComputeBoundMaterial()[0]
    )
    UsdGeom.Imageable(base).MakeInvisible()
