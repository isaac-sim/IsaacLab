# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Import-light checks for the Digital Twin conveyor playback scene."""

import math
import subprocess
import sys
from types import SimpleNamespace

import pytest

import isaaclab.sim as sim_utils

from isaaclab_tasks.contrib.conveyor_franka.conveyor_franka_asset_env_cfg import (
    _PRESENTATION_ASSETS,
    ConveyorFrankaA09A12EnvCfg,
    _presentation_layer,
)
from isaaclab_tasks.contrib.conveyor_franka.conveyor_franka_env_cfg import ConveyorFrankaEnvCfg
from isaaclab_tasks.contrib.conveyor_franka.conveyor_geometry import (
    BELT_CENTER_X,
    BELT_CENTER_Y,
    BELT_HALF_STRAIGHT,
    BELT_TOP_Z,
    BELT_TURN_RADIUS,
)


def test_a09_a12_config_import_does_not_preload_usd() -> None:
    """Task discovery must not import USD before a requested Kit application starts."""
    code = (
        "import sys; "
        "import isaaclab_tasks.contrib.conveyor_franka.conveyor_franka_asset_env_cfg; "
        "raise SystemExit('pxr' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], check=False)
    assert result.returncode == 0


def test_carton_visual_is_centered_on_the_original_40_mm_collider(tmp_path, monkeypatch) -> None:
    """Asset normalization changes appearance without moving or resizing the grasp surface."""
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    from isaaclab_tasks.contrib.conveyor_franka import conveyor_franka_asset_env_cfg as asset_cfg

    # Measured unscaled Cardbox_A1 bounds, used as an offline stand-in for the remote mesh.
    lower = (-0.34971755743026733, -0.260576993227005, 0.0)
    upper = (0.34971755743026733, 0.2605747878551483, 0.5099270939826965)
    source = Usd.Stage.CreateNew(str(tmp_path / "carton.usda"))
    root = UsdGeom.Xform.Define(source, "/Carton").GetPrim()
    source.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root)
    UsdPhysics.FixedJoint.Define(source, "/Carton/Joint")
    source.DefinePrim("/Carton/Graph", "OmniGraph")
    UsdPhysics.Scene.Define(source, "/Carton/PhysicsScene")
    mesh = UsdGeom.Cube.Define(source, "/Carton/Mesh")
    mesh.CreateSizeAttr(1.0)
    mesh.AddTranslateOp().Set(tuple((a + b) / 2 for a, b in zip(lower, upper)))
    mesh.AddScaleOp().Set(tuple(b - a for a, b in zip(lower, upper)))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.FilteredPairsAPI.Apply(mesh.GetPrim())
    mesh.GetPrim().AddAppliedSchema("PhysxCollisionAPI")
    support = UsdGeom.Mesh.Define(source, "/Carton/Support")
    support.CreatePointsAttr([(0.1, 0, 0.2), (0.1, 0.1, 0.255), (0, 0.1, 0.3)])
    support.CreateFaceVertexCountsAttr([3])
    support.CreateFaceVertexIndicesAttr([0, 1, 2])
    support.GetPrim().CreateAttribute("conveyor:supportAnchor", Sdf.ValueTypeNames.Double).Set(0.255)
    support.GetPrim().CreateAttribute("conveyor:supportScale", Sdf.ValueTypeNames.Double).Set(2.0)
    source.GetRootLayer().Save()
    monkeypatch.setattr(asset_cfg, "retrieve_file_path", lambda path: str(tmp_path / "carton.usda"))
    _presentation_layer.cache_clear()
    try:
        stage = sim_utils.create_new_stage()
        cfg = ConveyorFrankaA09A12EnvCfg().scene.cube_0.spawn
        cfg.parcel_usd_path = str(_PRESENTATION_ASSETS / "parcel_blue.usda")
        prim = cfg.func("/Cube", cfg)
        visual = stage.GetPrimAtPath("/Cube/CartonVisual")
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(visual)
        assert tuple(bounds.ComputeAlignedRange().GetMin()) == pytest.approx((-0.02,) * 3, abs=1e-8)
        assert tuple(bounds.ComputeAlignedRange().GetMax()) == pytest.approx((0.02,) * 3, abs=1e-8)
        extended = UsdGeom.Mesh(next(p for p in Usd.PrimRange(visual) if p.GetName() == "Support"))
        assert [p[2] for p in extended.GetPointsAttr().Get()] == pytest.approx((0.145, 0.255, 0.3))
        assert [p[2] for p in support.GetPointsAttr().Get()] == pytest.approx((0.2, 0.255, 0.3))
        assert prim.HasAPI(UsdPhysics.RigidBodyAPI)
        collider = stage.GetPrimAtPath("/Cube/geometry/mesh")
        assert collider.HasAPI(UsdPhysics.CollisionAPI)
        assert UsdGeom.Imageable(collider).ComputeVisibility() == "invisible"
        assert sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse()) == 1
        assert sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in stage.Traverse()) == 1
        imported = tuple(Usd.PrimRange(visual, Usd.TraverseInstanceProxies()))
        for api in (
            UsdPhysics.ArticulationRootAPI,
            UsdPhysics.RigidBodyAPI,
            UsdPhysics.MassAPI,
            UsdPhysics.CollisionAPI,
            UsdPhysics.MeshCollisionAPI,
            UsdPhysics.FilteredPairsAPI,
        ):
            assert not any(p.HasAPI(api) for p in imported)
        assert all("PhysxCollisionAPI" not in p.GetAppliedSchemas() for p in imported)
        # Inactive imported graphs, joints and scenes must not survive composition as executable prims.
        assert not any(p.GetTypeName() in {"PhysicsFixedJoint", "OmniGraph", "PhysicsScene"} for p in imported)
        # Resolving cached assets must leave the authored layer portable.
        assert all(
            path.startswith("https://")
            for path in Sdf.Layer.FindOrOpen(str(_PRESENTATION_ASSETS / "parcel.usda")).GetExternalReferences()
        )
    finally:
        _presentation_layer.cache_clear()


def test_asset_transforms_preserve_the_original_workcell() -> None:
    """The working straights and adjoining quarter-turns keep their original locations and radii."""
    from pxr import Sdf

    from isaaclab_tasks.contrib.conveyor_franka.conveyor_geometry import belt_collision_section_specs
    from isaaclab_tasks.contrib.conveyor_franka.conveyor_warehouse_geometry import (
        warehouse_belt_sections,
    )

    scene = ConveyorFrankaA09A12EnvCfg().scene
    layer = Sdf.Layer.FindOrOpen(scene.warehouse_visual.spawn.usd_path)
    scene._configure_route_assets()
    workspace_z = scene.ground.workspace_origin_offset[2]
    assert math.isclose(scene.robot.init_state.pos[2], workspace_z)
    assert math.isclose(scene.cube_0.init_state.pos[2], 0.28 + workspace_z)
    base_z = ConveyorFrankaEnvCfg().scene.conveyor_left_top_straight_collision.init_state.pos[2]
    assert math.isclose(scene.warehouse_left_section_0.init_state.pos[2], base_z + workspace_z)
    for side, sign, original_index in (("Left", 1, 1), ("Right", -1, 0)):
        sections = warehouse_belt_sections(side, velocity=0.35)
        original = belt_collision_section_specs(side)[original_index].geometry
        inner = sections[0].geometry
        assert inner.position == original.position
        assert inner.size == original.size
        visual = getattr(scene, f"conveyor_{side.lower()}_{'bottom' if sign > 0 else 'top'}_a09_visual")
        assert visual.init_state.pos[:2] == (
            BELT_CENTER_X + BELT_HALF_STRAIGHT,
            sign * (BELT_CENTER_Y - BELT_TURN_RADIUS),
        )
        assert 4 * visual.spawn.scale[0] == pytest.approx(2 * BELT_HALF_STRAIGHT)
        assert visual.init_state.pos[2] + 1.78053 * visual.spawn.scale[2] == pytest.approx(
            BELT_TOP_Z + scene.ground.workspace_origin_offset[2]
        )
        for name, x in (
            ("RightInner", BELT_CENTER_X + BELT_HALF_STRAIGHT),
            ("LeftInner", BELT_CENTER_X - BELT_HALF_STRAIGHT),
        ):
            turn = next(section.belt for section in sections if f"{name}Turn" in section.geometry.name)
            assert turn.pivot_point == (x, sign * BELT_CENTER_Y, 0)
            assert turn.radius == BELT_TURN_RADIUS
            authored = layer.GetPrimAtPath(f"/Warehouse/WorkcellConveyors/{side}{name}")
            position = authored.attributes["xformOp:translate"].default
            scale = authored.attributes["xformOp:scale"].default
            angle = math.radians(authored.attributes["xformOp:rotateZ"].default)
            # A03's original circular pivot is at (0, -1.4961); the referenced arc lands on the physical bend.
            radius = 1.4961 * scale[0]
            assert radius == pytest.approx(BELT_TURN_RADIUS)
            assert position[0] + radius * math.sin(angle) == pytest.approx(x)
            assert position[1] - radius * math.cos(angle) == pytest.approx(sign * BELT_CENTER_Y)


def test_warehouse_animation_uses_active_kit_viewer_and_policy_time(tmp_path, monkeypatch) -> None:
    """A CLI-selected Kit viewer animates and loops authored parcels without changing task state."""
    from pxr import Usd, UsdGeom

    from isaaclab_tasks.contrib.conveyor_franka.conveyor_franka_env import ConveyorFrankaEnv
    from isaaclab_tasks.contrib.conveyor_franka.conveyor_franka_warehouse_env import ConveyorFrankaWarehouseEnv

    path = str(tmp_path / "traffic.usda")
    source = Usd.Stage.CreateNew(path)
    root = UsdGeom.Xform.Define(source, "/Warehouse").GetPrim()
    source.SetDefaultPrim(root)
    source.SetTimeCodesPerSecond(60)
    source.SetEndTimeCode(120)
    parcel = UsdGeom.Xform.Define(source, "/Warehouse/Parcels/Parcel00")
    translate = parcel.AddTranslateOp()
    translate.Set((0, 0, 1), 0)
    translate.Set((2, 0, 1), 120)
    parcel.AddRotateXYZOp().Set((0, 0, 0))
    source.GetRootLayer().Save()
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/envs/env_0/WarehouseVisual").GetReferences().AddReference(path)
    callbacks = {}

    def initialize(env, cfg, **kwargs):
        env.cfg = cfg
        env.common_step_counter = 60
        env.scene = SimpleNamespace(env_prim_paths=["/World/envs/env_0"])
        env.sim = SimpleNamespace(
            stage=stage,
            visualizers=[SimpleNamespace(cfg=SimpleNamespace(visualizer_type="kit"))],
            set_setting=lambda name, value: None,
            add_render_callback=lambda name, callback: callbacks.update({name: callback}),
            remove_render_callback=lambda name: callbacks.pop(name, None),
        )

    monkeypatch.setattr(ConveyorFrankaEnv, "__init__", initialize)
    monkeypatch.setattr(ConveyorFrankaEnv, "close", lambda env: None)
    cfg = ConveyorFrankaA09A12EnvCfg()
    # CLI viewer selection is resolved by SimulationContext, not written back into this list.
    cfg.sim.visualizer_cfgs = []
    cfg.scene.warehouse_visual.spawn.usd_path = path
    env = ConveyorFrankaWarehouseEnv(cfg)
    try:
        callback = callbacks["conveyor_warehouse_animation"]
        callback(None)
        position = stage.GetPrimAtPath("/World/envs/env_0/WarehouseVisual/Parcels/Parcel00").GetAttribute(
            "xformOp:translate"
        )
        assert tuple(position.Get()) == pytest.approx((1, 0, 1))
        env.common_step_counter = 180
        callback(None)
        assert tuple(position.Get()) == pytest.approx((1, 0, 1))
    finally:
        env.close()
        _presentation_layer.cache_clear()
    assert not callbacks
