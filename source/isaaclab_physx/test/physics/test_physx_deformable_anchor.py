# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

from pxr import Usd, UsdGeom

_TET_POINTS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
_TET_INDICES = [(0, 1, 2, 3)]
_TRI_FACES = [0, 1, 2]


def _make_volume_body(stage, path="/World/Body"):
    body = UsdGeom.Xform.Define(stage, path).GetPrim()
    vis = UsdGeom.Mesh.Define(stage, f"{path}/mesh").GetPrim()
    vis.GetAttribute("points").Set(_TET_POINTS)
    vis.GetAttribute("faceVertexIndices").Set(_TRI_FACES)
    vis.GetAttribute("faceVertexCounts").Set([3])
    sim = UsdGeom.TetMesh.Define(stage, f"{path}/sim_mesh").GetPrim()
    sim.GetAttribute("points").Set(_TET_POINTS)
    sim.GetAttribute("tetVertexIndices").Set(_TET_INDICES)
    faces = UsdGeom.TetMesh.ComputeSurfaceFaces(UsdGeom.TetMesh(sim), Usd.TimeCode.Default())
    UsdGeom.TetMesh(sim).GetSurfaceFaceVertexIndicesAttr().Set(faces)
    return body, sim, vis


def test_physx_volume_anchor_applies_omni_apis_and_rest_state():
    from isaaclab_physx.physics import PhysxManager

    import isaaclab.sim as sim_utils

    sim_utils.create_new_stage()
    stage = sim_utils.get_current_stage()
    body, sim_mesh, vis_mesh = _make_volume_body(stage)
    PhysxManager.setup_deformable_body(body, "volume", sim_mesh, vis_mesh, stage)
    assert "OmniPhysicsDeformableBodyAPI" in body.GetPrimTypeInfo().GetAppliedAPISchemas()
    assert "OmniPhysicsVolumeDeformableSimAPI" in sim_mesh.GetPrimTypeInfo().GetAppliedAPISchemas()
    assert sim_mesh.GetAttribute("omniphysics:restShapePoints").Get() is not None
    assert sim_mesh.GetAttribute("omniphysics:restTetVtxIndices").Get() is not None
    assert vis_mesh.GetAttribute("deformablePose:default:omniphysics:points").Get() is not None
    assert list(sim_mesh.GetAttribute("deformablePose:default:omniphysics:purposes").Get()) == ["bindPose"]
