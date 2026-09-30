# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Import-light checks for the Digital Twin conveyor playback scene."""

import math
import subprocess
import sys
from types import SimpleNamespace

import gymnasium as gym
import pytest

import isaaclab.sim as sim_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.conveyor_franka.conveyor_franka_asset_env_cfg import (
    _PRESENTATION_ASSETS,
    ConveyorFrankaA09A12EnvCfg,
    _make_usd_subtree_visual_only,
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


def test_racetrack_and_sorting_tasks_share_the_policy_contract() -> None:
    """Sorting extends the four-cube task without changing its scene or policy configuration."""
    task = gym.spec("IsaacContrib-Conveyor-Franka-Newton-Play-v0")
    base_task = gym.spec("IsaacContrib-Conveyor-Franka-Newton-v0")
    base_cfg = ConveyorFrankaEnvCfg()
    base_scene = base_cfg.scene.to_dict()
    cfg = ConveyorFrankaA09A12EnvCfg()
    cfg.scene._configure_route_assets(cfg.commands.transfer.parcel_colors)

    assert task.kwargs["env_cfg_entry_point"].endswith(":ConveyorFrankaA09A12EnvCfg")
    assert base_task.kwargs["env_cfg_entry_point"].endswith(":ConveyorFrankaEnvCfg")
    assert task.kwargs["rsl_rl_cfg_entry_point"] == base_task.kwargs["rsl_rl_cfg_entry_point"]
    assert base_cfg.scene.to_dict() == base_scene
    assert [name for name in vars(base_cfg.scene) if name.startswith("cube_")] == [
        "cube_0",
        "cube_1",
        "cube_2",
        "cube_3",
    ]
    assert base_cfg.conveyor_force.transported_body_count_per_env == 4
    assert base_cfg.commands.transfer.class_type.__name__ == "ConveyorTransferCommand"
    assert cfg.commands.transfer.class_type.__name__ == "ConveyorSortCommand"
    assert cfg.conveyor_force.transported_body_count_per_env == 24
    assert cfg.scene.num_envs == 1
    assert cfg.actions == base_cfg.actions
    assert cfg.observations == base_cfg.observations
    assert cfg.commands.transfer.parcel_destinations == (0, 1) * 12
    assert cfg.commands.transfer.parcel_colors == ("blue", "orange", "green", "purple") * 6
    assert cfg.commands.transfer.randomize_arrivals is True
    assert cfg.commands.transfer.hold_steps == base_cfg.commands.transfer.hold_steps
    assert cfg.events == base_cfg.events
    assert cfg.rewards == base_cfg.rewards
    for name, term in vars(base_cfg.terminations).items():
        if name != "cube_out_of_workspace":
            assert getattr(cfg.terminations, name) == term
    assert cfg.terminations.cube_out_of_workspace.func == base_cfg.terminations.cube_out_of_workspace.func
    assert cfg.decimation == base_cfg.decimation
    assert cfg.sim.dt == base_cfg.sim.dt
    for cube_id in range(24):
        actual = getattr(cfg.scene, f"cube_{cube_id}").spawn.to_dict()
        expected = getattr(base_cfg.scene, "cube_0").spawn.to_dict()
        actual.pop("func")
        actual.pop("parcel_usd_path")
        expected.pop("func")
        assert actual == expected


def test_sorting_contacts_exclude_stationary_conveyor_pairs() -> None:
    """Touching belt sections must not consume contacts needed to support parcels."""
    import newton
    import warp as wp

    builder = newton.ModelBuilder()
    for x in (0.0, 0.15):
        section = newton.ModelBuilder()
        section.add_shape_box(-1, hx=0.1, hy=0.1, hz=0.1)
        builder.add_builder(section, xform=wp.transform((x, 0.0, 0.0), wp.quat_identity()))
    parcel = builder.add_body(xform=wp.transform((0.0, 0.0, 0.18), wp.quat_identity()))
    builder.add_shape_box(parcel, hx=0.1, hy=0.1, hz=0.1)
    model = builder.finalize(device="cpu")
    cfg = ConveyorFrankaA09A12EnvCfg().sim.physics.collision_cfg
    pipeline = newton.CollisionPipeline(model, **cfg.to_pipeline_args())
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)
    count = int(contacts.rigid_contact_count.numpy()[0])
    assert count > 0
    bodies = model.shape_body.numpy()
    first = bodies[contacts.rigid_contact_shape0.numpy()[:count]]
    second = bodies[contacts.rigid_contact_shape1.numpy()[:count]]
    assert ((first == parcel) | (second == parcel)).all()


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
    mesh = UsdGeom.Cube.Define(source, "/Carton/Mesh")
    mesh.CreateSizeAttr(1.0)
    mesh.AddTranslateOp().Set(tuple((a + b) / 2 for a, b in zip(lower, upper)))
    mesh.AddScaleOp().Set(tuple(b - a for a, b in zip(lower, upper)))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
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
        assert prim.HasAPI(UsdPhysics.RigidBodyAPI)
        collider = stage.GetPrimAtPath("/Cube/geometry/mesh")
        assert collider.HasAPI(UsdPhysics.CollisionAPI)
        assert UsdGeom.Imageable(collider).ComputeVisibility() == "invisible"
        assert sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse()) == 1
        assert sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in stage.Traverse()) == 1
        assert not any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(visual))
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


@pytest.mark.parametrize("use_slice", [False, True])
def test_warehouse_reset_loads_mixed_feeds_only_in_selected_environments(monkeypatch, use_slice) -> None:
    """A seeded reset shuffles all physical arrivals without disturbing other environments."""
    import torch

    from isaaclab_tasks.contrib.conveyor_franka.conveyor_cube_pool import ConveyorCubePool
    from isaaclab_tasks.contrib.conveyor_franka.conveyor_franka_env import ConveyorFrankaEnv
    from isaaclab_tasks.contrib.conveyor_franka.conveyor_franka_warehouse_env import ConveyorFrankaWarehouseEnv
    from isaaclab_tasks.contrib.conveyor_franka.conveyor_warehouse_geometry import warehouse_parcel_positions

    origins = torch.tensor([[0.0, 0.0, 0.8], [10.0, 12.0, 0.8]])
    cubes = {}
    before = []
    velocities = []
    positions = warehouse_parcel_positions()
    for i in range(len(positions)):
        pose = torch.tensor([[0.52, 0.27, 0.06, 0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
        pose[:, :3] += origins
        before.append(pose.clone())
        velocity = torch.ones(2, 6)
        velocities.append(velocity)

        def write(root_pose, env_ids, state=pose):
            state[env_ids] = root_pose

        def write_velocity(root_velocity, env_ids, state=velocity):
            state[env_ids] = root_velocity

        cubes[f"cube_{i}"] = SimpleNamespace(
            data=SimpleNamespace(root_pose_w=SimpleNamespace(torch=pose)),
            write_root_pose_to_sim_index=write,
            write_root_velocity_to_sim_index=write_velocity,
        )

    class Scene(dict):
        env_origins = origins
        _ALL_INDICES = torch.arange(2)

    env = ConveyorFrankaWarehouseEnv.__new__(ConveyorFrankaWarehouseEnv)
    env.scene = Scene(cubes)
    env.conveyor_cube_pool = ConveyorCubePool(tuple(cubes.values()), 2, "cpu")
    env._warehouse_animation = []
    env.sim = SimpleNamespace(device="cpu", remove_render_callback=lambda name: None)
    env.cfg = ConveyorFrankaA09A12EnvCfg()
    monkeypatch.setattr(ConveyorFrankaEnv, "_reset_idx", lambda self, ids: None)
    monkeypatch.setattr(ConveyorFrankaEnv, "close", lambda self: None)
    torch.manual_seed(42)
    env._reset_idx(slice(1, 2) if use_slice else torch.tensor([1]))
    actual_positions = []
    for i in range(len(positions)):
        actual = cubes[f"cube_{i}"].data.root_pose_w.torch
        torch.testing.assert_close(actual[0], before[i][0])
        torch.testing.assert_close(actual[1, 3:], before[i][1, 3:])
        actual_positions.append(actual[1, :3] - origins[1])
        torch.testing.assert_close(velocities[i][0], torch.ones(6))
        torch.testing.assert_close(velocities[i][1], torch.zeros(6))
    actual_positions = torch.stack(actual_positions)
    expected = torch.tensor(positions)
    distances = torch.linalg.vector_norm(actual_positions[:, None] - expected[None], dim=-1)
    assert distances.min(dim=1).values.max() < 1e-5
    assert distances.argmin(dim=1).unique().numel() == len(positions)
    assert not torch.allclose(actual_positions, expected)
    torch.manual_seed(42)
    env._reset_idx(slice(1, 2) if use_slice else torch.tensor([1]))
    repeated = torch.stack([cube.data.root_pose_w.torch[1, :3] - origins[1] for cube in cubes.values()])
    torch.testing.assert_close(repeated, actual_positions)


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


def test_warehouse_policy_view_preserves_local_states_and_physical_inventory():
    """Only remote transport is mapped to waiting slots; physical tensors remain untouched."""
    import torch

    from isaaclab_tasks.contrib.conveyor_franka.conveyor_franka_warehouse_env import ConveyorFrankaWarehouseEnv

    origin = torch.tensor([[10.0, 12.0, 0.8]])
    positions = (
        torch.tensor([[[0.52, 0.27, 0.06], [0.6, -0.1, 0.25], [2.8, 0.1, 0.56], [1.2, -0.59, 0.16]]]) + origin[:, None]
    )
    quaternions = torch.randn(1, 4, 4)
    velocities = torch.randn(1, 4, 6)
    original = tuple(value.clone() for value in (positions, quaternions, velocities))
    env = SimpleNamespace(
        scene=SimpleNamespace(env_origins=origin),
        device="cpu",
        cfg=SimpleNamespace(conveyor_force=SimpleNamespace(speed=0.35)),
        _in_workcell=ConveyorFrankaWarehouseEnv._in_workcell,
    )
    actual = ConveyorFrankaWarehouseEnv._adapt_policy_cube_state(env, positions, quaternions, velocities)
    for result, source, before in zip(actual, (positions, quaternions, velocities), original):
        torch.testing.assert_close(result[:, :2], source[:, :2])
        torch.testing.assert_close(source, before)
    torch.testing.assert_close(actual[0][0, 2:, 1:] - origin[0, 1:], torch.tensor([[0.75, 0.06], [-0.75, 0.06]]))
    torch.testing.assert_close(actual[1][0, 2:], torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]))


def test_warehouse_idle_parks_and_preserves_invalid_actions(monkeypatch):
    """Idle handling cannot mask invalid policy input or bypass an active transfer."""
    import torch

    from isaaclab_tasks.contrib.conveyor_franka.conveyor_franka_env import ConveyorFrankaEnv
    from isaaclab_tasks.contrib.conveyor_franka.conveyor_franka_warehouse_env import ConveyorFrankaWarehouseEnv

    class Scene(dict):
        num_envs = 3

    scene = Scene(
        robot=SimpleNamespace(
            data=SimpleNamespace(
                default_joint_pos=SimpleNamespace(torch=torch.zeros(3, 7)),
                joint_pos=SimpleNamespace(torch=torch.full((3, 7), 0.12)),
            )
        )
    )
    command = SimpleNamespace(has_target=torch.tensor([False, True, False]))
    env = ConveyorFrankaWarehouseEnv.__new__(ConveyorFrankaWarehouseEnv)
    env.scene = scene
    env.sim = SimpleNamespace(device="cpu", remove_render_callback=lambda name: None)
    env.cfg = SimpleNamespace(actions=SimpleNamespace(arm_action=SimpleNamespace(scale=0.12)))
    env.command_manager = SimpleNamespace(get_term=lambda name: command)
    env._warehouse_arm_joint_ids = list(range(7))
    env._warehouse_animation = []
    accepted = []
    result = ({"policy": torch.zeros(3, 123)}, None, None, None, {})

    def step(self, action):
        accepted.append(action)
        return result

    monkeypatch.setattr(ConveyorFrankaEnv, "step", step)
    monkeypatch.setattr(ConveyorFrankaEnv, "close", lambda self: None)
    action = torch.full((3, 8), 0.5)
    action[2, 0] = torch.nan
    assert env.step(action) is result
    actual = accepted[0]
    torch.testing.assert_close(actual[0], torch.tensor([-0.25] * 7 + [0.0]))
    torch.testing.assert_close(actual[1], action[1])
    torch.testing.assert_close(actual[2], action[2], equal_nan=True)


def test_parcel_pool_preserves_pinned_slots_and_eventually_assigns_every_physical_parcel():
    """Arrivals rotate through four distinct slots without displacing a held parcel or another environment."""
    import torch

    from isaaclab_tasks.contrib.conveyor_franka.conveyor_cube_pool import ConveyorCubePool

    pool = ConveyorCubePool((None,) * 16, 2, "cpu")
    pool.slot_ids[1] = torch.tensor([3, 2, 1, 0])
    positions = torch.zeros(2, 16, 3)
    positions[:, :, 0] = torch.linspace(0.4, 1.0, 16)
    local = torch.zeros(2, 16, dtype=torch.bool)
    local[0, 1] = True
    candidates = torch.zeros_like(local)
    candidates[0, 4:] = True
    target_slots = torch.tensor([0, 2])
    pinned = torch.tensor([True, True])
    initial_other_environment = pool.slot_ids[1].clone()
    for _ in range(6):
        changed = pool.refresh(positions, local, candidates, target_slots, pinned)
        assert changed.tolist() == [True, False]
        assert pool.slot_ids[0, :2].tolist() == [0, 1]
        assert len(pool.slot_ids[0].unique()) == 4
        torch.testing.assert_close(pool.slot_ids[1], initial_other_environment)
    assert bool((pool.assignment_counts[0] > 0).all())
    physical_id = int(pool.slot_ids[0, 2])
    pool.record_transfers(torch.tensor([0]), torch.tensor([2]))
    assert pool.transfer_counts[0, physical_id] == 1
    assert pool.transfer_counts.sum() == 1
    pool.reset(torch.tensor([0]))
    assert pool.slot_ids[0].tolist() == [0, 1, 2, 3]
    torch.testing.assert_close(pool.slot_ids[1], initial_other_environment)


def test_parcel_pool_gathers_per_environment_states_and_checks_unassigned_inventory():
    """All policy features use the same physical assignment; an unassigned fallen cube still terminates."""
    import torch

    from isaaclab_tasks.contrib.conveyor_franka.conveyor_cube_pool import ConveyorCubePool, cube_values
    from isaaclab_tasks.contrib.conveyor_franka.mdp.observations import _cube_state
    from isaaclab_tasks.contrib.conveyor_franka.mdp.terminations import cube_out_of_workspace

    assets = []
    for cube_id in range(6):
        values = {
            "root_pos_w": torch.tensor([[0.1 * cube_id, 0.27, 0.06], [10 + 0.1 * cube_id, 0.27, 0.06]]),
            "root_quat_w": torch.full((2, 4), float(cube_id)),
            "root_vel_w": torch.full((2, 6), float(cube_id)),
        }
        assets.append(
            SimpleNamespace(
                data=SimpleNamespace(**{name: SimpleNamespace(torch=value) for name, value in values.items()})
            )
        )
    pool = ConveyorCubePool(tuple(assets), 2, "cpu")
    pool.slot_ids[:] = torch.tensor([[4, 1, 2, 3], [0, 5, 2, 3]])
    env = SimpleNamespace(
        conveyor_cube_pool=pool, scene=SimpleNamespace(env_origins=torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
    )
    physical_before = cube_values(env, "root_pos_w", all_cubes=True).clone()
    states = _cube_state(env)
    for values, attribute in zip(states, ("root_pos_w", "root_quat_w", "root_vel_w")):
        for row in range(2):
            expected = torch.stack([getattr(assets[i].data, attribute).torch[row] for i in pool.slot_ids[row]])
            torch.testing.assert_close(values[row], expected)
    torch.testing.assert_close(cube_values(env, "root_pos_w", all_cubes=True), physical_before)
    assert not cube_out_of_workspace(env).any()
    assets[5].data.root_pos_w.torch[0, 2] = -1.0  # Parcel 5 is not assigned in environment 0.
    assert cube_out_of_workspace(env).tolist() == [True, False]
