# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import contextlib
from types import SimpleNamespace

import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.physics.scene_data_requirements import SceneDataRequirement
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import build_simulation_context
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration."""

    # articulation
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/SimpleArticulation/revolute_articulation.usd",
        ),
        actuators={
            "joint": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=100.0, damping=1.0),
        },
    )
    # rigid object
    rigid_obj = RigidObjectCfg(
        prim_path="/World/envs/env_.*/RigidObj",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
        ),
    )


@pytest.fixture
def setup_scene(request):
    """Create simulation context with the specified device."""
    device = request.getfixturevalue("device")
    with build_simulation_context(device=device, auto_add_lighting=True, add_ground_plane=True) as sim:
        sim._app_control_on_stop_handle = None

        def make_scene(num_envs: int, env_spacing: float = 1.0):
            scene_cfg = MySceneCfg(num_envs=num_envs, env_spacing=env_spacing)
            return scene_cfg

        yield make_scene, sim
    # Note: cleanup is handled by build_simulation_context's finally block


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_relative_flag(device, setup_scene):
    make_scene, sim = setup_scene
    scene_cfg = make_scene(num_envs=4)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # test relative == False produces different result than relative == True
    assert_state_different(scene.get_state(is_relative=False), scene.get_state(is_relative=True))

    # test is relative == False
    prev_state = scene.get_state(is_relative=False)
    joint_pos = torch.rand_like(scene["robot"].data.joint_pos.torch)
    joint_vel = torch.rand_like(scene["robot"].data.joint_pos.torch)
    scene["robot"].write_joint_position_to_sim_index(position=joint_pos)
    scene["robot"].write_joint_velocity_to_sim_index(velocity=joint_vel)
    next_state = scene.get_state(is_relative=False)
    assert_state_different(prev_state, next_state)
    scene.reset_to(prev_state, is_relative=False)
    assert_state_equal(prev_state, scene.get_state(is_relative=False))

    # test is relative == True
    prev_state = scene.get_state(is_relative=True)
    joint_pos = torch.rand_like(scene["robot"].data.joint_pos.torch)
    joint_vel = torch.rand_like(scene["robot"].data.joint_pos.torch)
    scene["robot"].write_joint_position_to_sim_index(position=joint_pos)
    scene["robot"].write_joint_velocity_to_sim_index(velocity=joint_vel)
    next_state = scene.get_state(is_relative=True)
    assert_state_different(prev_state, next_state)
    scene.reset_to(prev_state, is_relative=True)
    assert_state_equal(prev_state, scene.get_state(is_relative=True))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_reset_to_env_ids_input_types(device, setup_scene):
    make_scene, sim = setup_scene
    scene_cfg = make_scene(num_envs=4)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # test env_ids = None
    prev_state = scene.get_state()
    joint_pos = torch.rand_like(scene["robot"].data.joint_pos.torch)
    joint_vel = torch.rand_like(scene["robot"].data.joint_pos.torch)
    scene["robot"].write_joint_position_to_sim_index(position=joint_pos)
    scene["robot"].write_joint_velocity_to_sim_index(velocity=joint_vel)
    scene.reset_to(prev_state, env_ids=None)
    assert_state_equal(prev_state, scene.get_state())

    # test env_ids = torch tensor
    joint_pos = torch.rand_like(scene["robot"].data.joint_pos.torch)
    joint_vel = torch.rand_like(scene["robot"].data.joint_pos.torch)
    scene["robot"].write_joint_position_to_sim_index(position=joint_pos)
    scene["robot"].write_joint_velocity_to_sim_index(velocity=joint_vel)
    scene.reset_to(prev_state, env_ids=torch.arange(scene.num_envs, device=scene.device, dtype=torch.int32))
    assert_state_equal(prev_state, scene.get_state())


def test_clone_environments_executes_env_root_plan_with_positions(monkeypatch: pytest.MonkeyPatch):
    """Env-root plans replicate the whole environment and keep grid positions."""
    from isaaclab.cloner import ClonePlan

    scene = object.__new__(InteractiveScene)
    scene.cfg = SimpleNamespace(replicate_physics=False, num_envs=3)
    scene.stage = object()
    scene.physics_backend = "physx"
    scene._sensors = {}

    set_plan_calls: list = []
    sim_state: dict = {"plan": None}

    def _set_clone_plan(plan):
        sim_state["plan"] = plan
        set_plan_calls.append(plan)

    scene.sim = SimpleNamespace(
        get_scene_data_requirements=lambda: SceneDataRequirement(),
        update_scene_data_requirements=lambda requirements: None,
        set_clone_plan=_set_clone_plan,
        get_clone_plan=lambda: sim_state["plan"],
    )
    scene.env_fmt = "/World/envs/env_{}"
    scene._ALL_INDICES = torch.arange(3, dtype=torch.long)
    scene._default_env_pose = torch.zeros((3, 7), dtype=torch.float32)
    scene._default_env_pose[:, 6] = 1.0  # identity quaternion (xyzw)
    scene._clone_plan = ClonePlan(
        sources=(scene.env_fmt.format(0),),
        destinations=(scene.env_fmt,),
        clone_mask=torch.ones((1, scene.num_envs), dtype=torch.bool),
    )
    # Avoid binding this unit test to global SimulationContext singleton state.
    monkeypatch.setattr(InteractiveScene, "device", property(lambda self: "cpu"))

    # ``disabled_fabric_change_notifies`` resolves the stage via UsdUtils.StageCache and would
    # crash on the bare ``object()`` mocked above. This unit test exercises clone-dispatch
    # logic only; the fabric notice path has its own coverage in ``test_cloner.py``.
    @contextlib.contextmanager
    def _noop_fabric_notices(stage, *, restore=True):
        yield

    monkeypatch.setattr("isaaclab.scene.interactive_scene.cloner.disabled_fabric_change_notifies", _noop_fabric_notices)

    physics_calls = []
    usd_calls = []

    def _physics_clone_fn(stage, *args, **kwargs):
        physics_calls.append((stage, args, kwargs))

    def _usd_replicate(stage, *args, **kwargs):
        usd_calls.append((stage, args, kwargs))

    scene.cloner_cfg = SimpleNamespace(
        device="cpu",
        physics_clone_fn=_physics_clone_fn,
        clone_usd=True,
    )
    monkeypatch.setattr("isaaclab.scene.interactive_scene.cloner.usd_replicate", _usd_replicate)

    scene.clone_environments(copy_from_source=False)
    assert len(physics_calls) == 1
    assert len(usd_calls) == 1
    mapping = physics_calls[0][1][3]
    assert mapping.dtype == torch.bool
    assert mapping.shape == (1, scene.num_envs)
    # Positions are a zero-copy ``[:, :3]`` view of ``_default_env_pose``: same storage, sliced shape.
    pose_storage_ptr = scene._default_env_pose.untyped_storage().data_ptr()
    physics_positions = physics_calls[0][2]["positions"]
    usd_positions = usd_calls[0][2]["positions"]
    assert physics_positions.untyped_storage().data_ptr() == pose_storage_ptr
    assert physics_positions.shape == (scene.num_envs, 3)
    assert usd_positions.untyped_storage().data_ptr() == pose_storage_ptr
    assert usd_positions.shape == (scene.num_envs, 3)
    assert len(set_plan_calls) == 1
    plan = set_plan_calls[-1]
    assert isinstance(plan, ClonePlan)
    assert plan.sources == (scene.env_fmt.format(0),)
    assert plan.destinations == (scene.env_fmt,)
    assert plan.clone_mask.shape == (1, scene.num_envs)
    assert scene.clone_plan is plan

    physics_calls.clear()
    usd_calls.clear()
    set_plan_calls.clear()
    scene.clone_environments(copy_from_source=True)
    assert len(physics_calls) == 0
    assert len(usd_calls) == 1
    assert len(set_plan_calls) == 1


def test_clone_environments_skips_replication_without_plan():
    """Direct-path cfg scenes publish no plan and do not dispatch cloners."""
    scene = object.__new__(InteractiveScene)
    scene._clone_plan = None
    set_plan_calls = []
    scene.sim = SimpleNamespace(set_clone_plan=set_plan_calls.append)

    scene.clone_environments(copy_from_source=False)

    assert set_plan_calls == [None]


def test_clone_environments_executes_asset_level_plan_without_usd_positions(monkeypatch: pytest.MonkeyPatch):
    """Asset-level plans preserve env-root transforms by skipping USD positions."""
    from isaaclab.cloner import ClonePlan

    scene = object.__new__(InteractiveScene)
    scene.cfg = SimpleNamespace(replicate_physics=False, num_envs=2)
    scene.stage = object()
    scene.physics_backend = "physx"
    scene._sensors = {}
    scene.env_fmt = "/World/envs/env_{}"
    scene._ALL_INDICES = torch.arange(2, dtype=torch.long)
    scene._default_env_pose = torch.ones((2, 7), dtype=torch.float32)
    scene._default_env_pose[:, 3:6] = 0.0  # identity quaternion (xyzw)
    scene._clone_plan = ClonePlan(
        sources=("/World/envs/env_0/Object", "/World/envs/env_1/Object"),
        destinations=("/World/envs/env_{}/Object", "/World/envs/env_{}/Object"),
        clone_mask=torch.tensor([[True, False], [False, True]], dtype=torch.bool),
    )

    set_plan_calls: list = []
    scene.sim = SimpleNamespace(set_clone_plan=set_plan_calls.append)
    monkeypatch.setattr(InteractiveScene, "device", property(lambda self: "cpu"))

    @contextlib.contextmanager
    def _noop_fabric_notices(stage, *, restore=True):
        yield

    monkeypatch.setattr("isaaclab.scene.interactive_scene.cloner.disabled_fabric_change_notifies", _noop_fabric_notices)
    monkeypatch.setattr(
        "isaaclab.scene.interactive_scene.cloner.usd_replicate",
        lambda *args, **kwargs: usd_calls.append((args, kwargs)),
    )

    physics_calls = []
    usd_calls = []
    scene.cloner_cfg = SimpleNamespace(
        device="cpu",
        physics_clone_fn=lambda *args, **kwargs: physics_calls.append((args, kwargs)),
        clone_usd=True,
    )

    scene.clone_environments(copy_from_source=False)

    assert len(physics_calls) == 1
    physics_positions = physics_calls[0][1]["positions"]
    assert physics_positions.untyped_storage().data_ptr() == scene._default_env_pose.untyped_storage().data_ptr()
    assert physics_positions.shape == (scene.num_envs, 3)
    assert len(usd_calls) == 1
    assert usd_calls[0][1]["positions"] is None
    assert set_plan_calls == [scene._clone_plan]


def test_build_clone_plan_from_cfg_plans_multi_and_single_spawners(monkeypatch: pytest.MonkeyPatch):
    """Heterogeneous planning writes source paths for multi and single spawners."""
    from isaaclab.cloner import sequential

    scene = object.__new__(InteractiveScene)
    scene.cfg = SimpleNamespace(
        num_envs=4,
        object=SimpleNamespace(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.MultiAssetSpawnerCfg(
                assets_cfg=[
                    sim_utils.ConeCfg(radius=0.1, height=0.2),
                    sim_utils.SphereCfg(radius=0.1),
                ]
            ),
        ),
        robot=SimpleNamespace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
        ),
    )
    scene.env_fmt = "/World/envs/env_{}"
    scene.cloner_cfg = SimpleNamespace(clone_strategy=sequential)
    monkeypatch.setattr(InteractiveScene, "device", property(lambda self: "cpu"))

    plan = scene._build_clone_plan_from_cfg()

    assert plan is not None
    assert plan.sources == (
        "/World/envs/env_0/Object",
        "/World/envs/env_1/Object",
        "/World/envs/env_0/Robot",
    )
    assert plan.destinations == (
        "/World/envs/env_{}/Object",
        "/World/envs/env_{}/Object",
        "/World/envs/env_{}/Robot",
    )
    assert scene.cfg.object.spawn.spawn_paths == ["/World/envs/env_0/Object", "/World/envs/env_1/Object"]
    assert scene.cfg.robot.spawn.spawn_path == "/World/envs/env_0/Robot"
    assert scene.cfg.object.prim_path == "{ENV_REGEX_NS}/Object"
    assert scene.cfg.robot.prim_path == "{ENV_REGEX_NS}/Robot"
    assert torch.equal(plan.clone_mask.to(torch.int).argmax(dim=0).cpu(), torch.tensor([0, 1, 0, 1]))


def test_build_clone_plan_from_cfg_defaults_to_env0_plan(monkeypatch: pytest.MonkeyPatch):
    """Homogeneous cfg scenes use the default env_0-to-all ClonePlan."""
    from isaaclab.cloner import sequential

    scene = object.__new__(InteractiveScene)
    scene.cfg = SimpleNamespace(
        num_envs=3,
        robot=SimpleNamespace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
        ),
    )
    scene.env_fmt = "/World/envs/env_{}"
    scene.cloner_cfg = SimpleNamespace(clone_strategy=sequential)
    monkeypatch.setattr(InteractiveScene, "device", property(lambda self: "cpu"))

    plan = scene._build_clone_plan_from_cfg()

    assert plan is not None
    assert plan.sources == ("/World/envs/env_0",)
    assert plan.destinations == (scene.env_fmt,)
    assert plan.clone_mask.shape == (1, scene.num_envs)
    assert scene.cfg.robot.spawn.spawn_path == "/World/envs/env_0/Robot"


def test_build_clone_plan_from_cfg_returns_none_without_env_scoped_groups(monkeypatch: pytest.MonkeyPatch):
    """Direct-path cfg scenes should not force env-root replication."""
    from isaaclab.cloner import sequential

    scene = object.__new__(InteractiveScene)
    scene.cfg = SimpleNamespace(
        num_envs=1,
        robot=SimpleNamespace(
            prim_path="/World/Robot",
            spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
        ),
    )
    scene.env_fmt = "/World/envs/env_{}"
    scene.cloner_cfg = SimpleNamespace(clone_strategy=sequential)
    monkeypatch.setattr(InteractiveScene, "device", property(lambda self: "cpu"))

    assert scene._build_clone_plan_from_cfg() is None
    assert scene.cfg.robot.spawn.spawn_path is None


def test_build_clone_plan_from_cfg_sets_collection_member_paths(monkeypatch: pytest.MonkeyPatch):
    """Rigid object collection members are planned independently."""
    from isaaclab.cloner import sequential

    scene = object.__new__(InteractiveScene)
    cube_cfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
    )
    shape_cfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shape",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[sim_utils.ConeCfg(radius=0.1, height=0.2), sim_utils.SphereCfg(radius=0.1)]
        ),
    )
    scene.cfg = SimpleNamespace(
        num_envs=4,
        objects=RigidObjectCollectionCfg(rigid_objects={"cube": cube_cfg, "shape": shape_cfg}),
    )
    scene.env_fmt = "/World/envs/env_{}"
    scene.cloner_cfg = SimpleNamespace(clone_strategy=sequential)
    monkeypatch.setattr(InteractiveScene, "device", property(lambda self: "cpu"))

    plan = scene._build_clone_plan_from_cfg()

    assert plan is not None
    planned_cube = scene.cfg.objects.rigid_objects["cube"]
    planned_shape = scene.cfg.objects.rigid_objects["shape"]
    assert planned_cube.spawn.spawn_path == "/World/envs/env_0/Cube"
    assert planned_shape.spawn.spawn_paths == ["/World/envs/env_0/Shape", "/World/envs/env_1/Shape"]
    assert "/World/envs/env_{}/Cube" in plan.destinations
    assert "/World/envs/env_{}/Shape" in plan.destinations


def test_build_clone_plan_from_cfg_marks_unused_variants(monkeypatch: pytest.MonkeyPatch):
    """Unused variants keep a mask row but do not get spawned."""
    from isaaclab.cloner import sequential

    scene = object.__new__(InteractiveScene)
    scene.cfg = SimpleNamespace(
        num_envs=2,
        object=SimpleNamespace(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.MultiAssetSpawnerCfg(
                assets_cfg=[
                    sim_utils.ConeCfg(radius=0.1, height=0.2),
                    sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
                    sim_utils.SphereCfg(radius=0.1),
                ]
            ),
        ),
    )
    scene.env_fmt = "/World/envs/env_{}"
    scene.cloner_cfg = SimpleNamespace(clone_strategy=sequential)
    monkeypatch.setattr(InteractiveScene, "device", property(lambda self: "cpu"))

    plan = scene._build_clone_plan_from_cfg()

    assert plan is not None
    assert scene.cfg.object.spawn.spawn_paths == ["/World/envs/env_0/Object", "/World/envs/env_1/Object", None]
    assert plan.clone_mask[2].sum() == 0


def test_aggregate_scene_data_requirements_merges_visualizers_and_renderers(monkeypatch: pytest.MonkeyPatch):
    """Scene aggregation must OR visualizer and sensor-renderer requirements onto sim context.

    Replaces the old test that asserted a clone-time visualizer hook was installed from
    requirements. The hook is gone; the only remaining behavior is publishing the merged
    :class:`SceneDataRequirement` to the simulation context.
    """
    scene = object.__new__(InteractiveScene)
    scene.physics_backend = "physx"
    scene.stage = object()
    scene._sensors = {
        "cam": SimpleNamespace(cfg=SimpleNamespace(renderer_cfg=SimpleNamespace(renderer_type="newton_warp")))
    }

    posted: list = []
    scene.sim = SimpleNamespace(
        get_scene_data_requirements=lambda: SceneDataRequirement(),
        update_scene_data_requirements=posted.append,
    )

    scene._aggregate_scene_data_requirements({"rerun"})

    assert len(posted) == 1
    merged = posted[0]
    assert merged.requires_newton_model


def assert_state_equal(s1: dict, s2: dict, path=""):
    """
    Recursively assert that s1 and s2 have the same nested keys
    and that every tensor leaf is exactly equal.
    """
    assert set(s1.keys()) == set(s2.keys()), f"Key mismatch at {path}: {s1.keys()} vs {s2.keys()}"
    for k in s1:
        v1, v2 = s1[k], s2[k]
        subpath = f"{path}.{k}" if path else k
        if isinstance(v1, dict):
            assert isinstance(v2, dict), f"Type mismatch at {subpath}"
            assert_state_equal(v1, v2, path=subpath)
        else:
            # leaf: should be a torch.Tensor
            assert isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor), f"Expected tensors at {subpath}"
            if not torch.equal(v1, v2):
                diff = (v1 - v2).abs().max()
                pytest.fail(f"Tensor mismatch at {subpath}, max abs diff = {diff}")


def assert_state_different(s1: dict, s2: dict, path=""):
    """
    Recursively scan s1 and s2 (which must have identical keys) and
    succeed as soon as you find one tensor leaf that differs.
    If you reach the end with everything equal, fail the test.
    """
    assert set(s1.keys()) == set(s2.keys()), f"Key mismatch at {path}: {s1.keys()} vs {s2.keys()}"
    for k in s1:
        v1, v2 = s1[k], s2[k]
        subpath = f"{path}.{k}" if path else k
        if isinstance(v1, dict):
            # recurse; if any nested call returns (i.e. finds a diff), we propagate success
            try:
                assert_state_different(v1, v2, path=subpath)
                return
            except AssertionError:
                continue
        else:
            assert isinstance(v1, torch.Tensor) and isinstance(v2, torch.Tensor), f"Expected tensors at {subpath}"
            if not torch.equal(v1, v2):
                return  # found a difference → success
    pytest.fail(f"No differing tensor found in nested state at {path}")
