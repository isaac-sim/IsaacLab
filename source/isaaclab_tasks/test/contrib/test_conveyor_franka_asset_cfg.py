# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Import-light checks for the Digital Twin conveyor playback scene."""

import math
import subprocess
import sys

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.conveyor_franka.conveyor_franka_asset_env_cfg import (
    ConveyorFrankaA09A12EnvCfg,
    _make_usd_subtree_visual_only,
)
from isaaclab_tasks.contrib.conveyor_franka.conveyor_franka_env_cfg import ConveyorFrankaEnvCfg
from isaaclab_tasks.contrib.conveyor_franka.conveyor_geometry import (
    BELT_CENTER_X,
    BELT_CENTER_Y,
    BELT_HALF_STRAIGHT,
    BELT_TOP_Z,
    BELT_TURN_RADIUS,
)


def test_a09_a12_play_task_reuses_the_newton_policy_contract() -> None:
    """The visual task is registered as a Play variant with unchanged policy-facing config."""
    task = gym.spec("IsaacContrib-Conveyor-Franka-Newton-Play-v0")
    cfg = ConveyorFrankaA09A12EnvCfg()
    base_cfg = ConveyorFrankaEnvCfg()

    assert task.kwargs["env_cfg_entry_point"].endswith(":ConveyorFrankaA09A12EnvCfg")
    assert task.id.replace("-Play", "") == "IsaacContrib-Conveyor-Franka-Newton-v0"
    assert cfg.scene.num_envs == 1
    assert cfg.actions == base_cfg.actions
    assert cfg.observations == base_cfg.observations
    assert cfg.commands == base_cfg.commands
    assert cfg.events == base_cfg.events
    assert cfg.rewards == base_cfg.rewards
    assert cfg.terminations == base_cfg.terminations
    assert cfg.decimation == base_cfg.decimation
    assert cfg.sim.dt == base_cfg.sim.dt


def test_a09_a12_config_import_does_not_preload_usd() -> None:
    """Task discovery must not import USD before a requested Kit application starts."""
    code = (
        "import sys; "
        "import isaaclab_tasks.contrib.conveyor_franka.conveyor_franka_asset_env_cfg; "
        "raise SystemExit('pxr' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], check=False)
    assert result.returncode == 0


def test_visual_only_usd_strips_physics_and_execution_metadata() -> None:
    """Decorative references cannot introduce bodies, contacts, joints, or action graphs."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/VisualAsset", "Xform")
    body = stage.DefinePrim("/VisualAsset/Body", "Xform")
    shape = stage.DefinePrim("/VisualAsset/Body/Shape", "Cube")
    joint = UsdPhysics.FixedJoint.Define(stage, "/VisualAsset/Joint").GetPrim()
    graph = stage.DefinePrim("/VisualAsset/ActionGraph", "OmniGraph")
    physics_scene = UsdPhysics.Scene.Define(stage, "/VisualAsset/PhysicsScene").GetPrim()

    UsdPhysics.ArticulationRootAPI.Apply(root)
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.MassAPI.Apply(body)
    UsdPhysics.CollisionAPI.Apply(shape)
    UsdPhysics.MeshCollisionAPI.Apply(shape)
    UsdPhysics.FilteredPairsAPI.Apply(shape)
    shape.AddAppliedSchema("PhysxCollisionAPI")

    _make_usd_subtree_visual_only(root)

    assert not root.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert not body.HasAPI(UsdPhysics.RigidBodyAPI)
    assert not body.HasAPI(UsdPhysics.MassAPI)
    assert not shape.HasAPI(UsdPhysics.CollisionAPI)
    assert not shape.HasAPI(UsdPhysics.MeshCollisionAPI)
    assert not shape.HasAPI(UsdPhysics.FilteredPairsAPI)
    assert "PhysxCollisionAPI" not in shape.GetAppliedSchemas()
    assert not joint.IsActive()
    assert not graph.IsActive()
    assert not physics_scene.IsActive()


def test_digital_twin_assets_replace_only_procedural_render_geometry() -> None:
    """A09/A12 visuals coexist with the unchanged lightweight collision proxies."""
    scene = ConveyorFrankaA09A12EnvCfg().scene

    assert scene.conveyor_left_belt_visual is None
    assert scene.guard_left_inner_visual is None
    assert hasattr(scene, "conveyor_left_top_straight_collision")
    assert hasattr(scene, "conveyor_left_right_turn_collision")
    assert hasattr(scene, "guard_left_inner_collision")

    asset_names = tuple(name for name in vars(scene) if name.endswith(("_a09_visual", "_a12_visual")))
    assert len(asset_names) == 8
    assert sum(name.endswith("_a09_visual") for name in asset_names) == 4
    assert sum(name.endswith("_a12_visual") for name in asset_names) == 4

    for name in asset_names:
        asset = getattr(scene, name)
        assert asset.spawn.usd_path.endswith("ConveyorBelt_A09.usd" if "a09" in name else "ConveyorBelt_A12.usd")
        assert asset.spawn.collision_props is None
        assert asset.spawn.make_uninstanceable


def test_thor_table_is_visual_only_and_all_support_feet_reach_the_ground() -> None:
    """The Thor mount and measured conveyor lows sit on the common z=0 floor."""
    cfg = ConveyorFrankaA09A12EnvCfg()
    scene = cfg.scene

    assert scene.tabletop.prim_path.endswith("/RobotThorTableVisual")
    assert scene.tabletop.spawn.usd_path.endswith("/Props/Mounts/thor_table.usd")
    assert scene.tabletop.spawn.collision_props is None
    ground_z = 0.0
    assert scene.table_pedestal is None
    assert math.isclose(scene.tabletop.init_state.pos[2] - 0.795 * scene.tabletop.spawn.scale[2], ground_z)

    for name in vars(scene):
        if name.endswith(("_a09_visual", "_a12_visual")):
            assert math.isclose(getattr(scene, name).init_state.pos[2], ground_z)

    assert ground_z == 0.0
    workspace_z = scene.ground.workspace_origin_offset[2]
    assert 0.2 < workspace_z < 0.3
    assert math.isclose(scene.robot.init_state.pos[2], workspace_z)
    assert math.isclose(scene.cube_0.init_state.pos[2], 0.06 + workspace_z)
    base_collision_z = ConveyorFrankaEnvCfg().scene.conveyor_left_top_straight_collision.init_state.pos[2]
    assert math.isclose(scene.conveyor_left_top_straight_collision.init_state.pos[2], base_collision_z + workspace_z)


def test_warehouse_props_are_render_only_and_pallet_bays_do_not_overlap() -> None:
    """Scene dressing stays outside the policy contract and owns no collision."""
    scene = ConveyorFrankaA09A12EnvCfg().scene
    assert not any(name.startswith("sorter_") for name in vars(scene))

    prop_names = ("packing_station_visual", "loaded_pallet_visual", "empty_pallet_visual")
    for name in prop_names:
        asset = getattr(scene, name)
        assert asset.spawn.collision_props is None
        assert asset.spawn.make_uninstanceable

    # Measured, conservatively axis-aligned extents after scaling leave a clear
    # aisle between the loaded and empty pallet bays even with their rotations.
    loaded = scene.loaded_pallet_visual
    empty = scene.empty_pallet_visual
    loaded_half_extent_x = (
        0.5 * loaded.spawn.scale[0] * (1.203 * math.cos(math.radians(30.0)) + 0.80281 * math.sin(math.radians(30.0)))
    )
    empty_half_extent_x = (
        0.5 * empty.spawn.scale[0] * (1.213235 * math.cos(math.radians(15.0)) + 0.802298 * math.sin(math.radians(15.0)))
    )
    assert loaded.init_state.pos[0] + loaded_half_extent_x < empty.init_state.pos[0] - empty_half_extent_x

    assert scene.ground.terrain_type == "plane"
    assert scene.ground.visual_material.diffuse_color == (0.055, 0.065, 0.082)
    assert scene.warehouse_back_wall_visual.spawn.collision_props is None
    assert scene.warehouse_side_wall_visual.spawn.collision_props is None
    assert scene.safety_zone_0_visual.spawn.collision_props is None


def test_asset_transforms_match_the_existing_racetrack_surface() -> None:
    """Asset endpoints, radius, and belt crown align with the policy's original geometry."""
    scene = ConveyorFrankaA09A12EnvCfg().scene
    top = scene.conveyor_left_top_a09_visual
    right_turn = scene.conveyor_left_right_a12_visual
    left_turn = scene.conveyor_left_left_a12_visual

    assert top.init_state.pos[:2] == (BELT_CENTER_X + BELT_HALF_STRAIGHT, BELT_CENTER_Y + BELT_TURN_RADIUS)
    assert right_turn.init_state.pos[:2] == top.init_state.pos[:2]
    assert left_turn.init_state.pos[:2] == (
        BELT_CENTER_X - BELT_HALF_STRAIGHT,
        BELT_CENTER_Y - BELT_TURN_RADIUS,
    )
    assert left_turn.init_state.rot == (0.0, 0.0, 1.0, 0.0)

    a09_length = 4.0 * top.spawn.scale[0]
    a12_diameter = 2.9922 * right_turn.spawn.scale[0]
    asset_belt_top = right_turn.init_state.pos[2] + 1.78053 * right_turn.spawn.scale[2]
    assert math.isclose(a09_length, 2.0 * BELT_HALF_STRAIGHT)
    assert math.isclose(a12_diameter, 2.0 * BELT_TURN_RADIUS)
    assert math.isclose(right_turn.spawn.scale[2], right_turn.spawn.scale[1])
    assert math.isclose(asset_belt_top, BELT_TOP_Z + scene.ground.workspace_origin_offset[2])
