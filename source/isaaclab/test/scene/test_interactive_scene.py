# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

from types import SimpleNamespace

import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.cloner import CloneCfg, InclusionSet, make_clone_plan, sequential
from isaaclab.physics.scene_data_requirements import SceneDataRequirement
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg, Selector, SelectorCfg
from isaaclab.sim import build_simulation_context
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass


def _make_selector(num_envs: int) -> Selector:
    return Selector(SelectorCfg(), num_envs=num_envs, device="cpu")


def _prepare_unit_scene(monkeypatch: pytest.MonkeyPatch, num_envs: int, **assets) -> InteractiveScene:
    scene = object.__new__(InteractiveScene)
    scene.cfg = SimpleNamespace(num_envs=num_envs, clone_cfg=CloneCfg(), selector_cfg=SelectorCfg(), **assets)
    scene.cloner_cfg = CloneCfg(clone_strategy=sequential)
    scene._env_ns = "/World/envs"
    scene._env_regex_ns = "/World/envs/env_.*"
    scene._selector = _make_selector(num_envs)
    scene._inactive_selector_assets = set()
    monkeypatch.setattr(InteractiveScene, "device", property(lambda self: "cpu"))
    return scene


def _build_unit_plan(scene: InteractiveScene):
    cfgs = scene._collect_asset_cfgs()
    plan = make_clone_plan(
        cfgs,
        num_clones=scene.cfg.num_envs,
        env_spacing=1.0,
        device="cpu",
        clone_strategy=scene.cloner_cfg.clone_strategy,
        valid_set=scene._clone_valid_set,
    )
    scene._apply_selector_from_clone_plan(plan)
    return plan


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


@configclass
class _HeteroStaticSceneCfg(InteractiveSceneCfg):
    """Two task groups where only group A also carries a static (non-physics) asset."""

    clone_cfg = CloneCfg(
        clone_strategy=sequential,
        clone_combinations=[
            InclusionSet(assets=["robot_a", "table_a"], weight=1),
            InclusionSet(assets=["robot_b"], weight=1),
        ],
    )
    robot_a = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot_A",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/SimpleArticulation/revolute_articulation.usd",
        ),
        actuators={"joint": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=100.0, damping=1.0)},
    )
    robot_b = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot_B",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/SimpleArticulation/revolute_articulation.usd",
        ),
        actuators={"joint": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=100.0, damping=1.0)},
    )
    table_a = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table_A",
        spawn=sim_utils.CuboidCfg(size=(0.2, 0.2, 0.2)),
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_heterogeneous_static_asset_cloned_into_group_envs(device):
    """A static ``AssetBaseCfg`` asset is cloned into every env of its group, not only its source env.

    Static assets register no backend physics view, so the scene must queue their USD replication
    explicitly; otherwise heterogeneous (per-asset clone-row) scenes leave them in just one env.
    """
    with build_simulation_context(device=device, auto_add_lighting=True, add_ground_plane=True) as sim:
        sim._app_control_on_stop_handle = None
        InteractiveScene(_HeteroStaticSceneCfg(num_envs=4, env_spacing=2.0))
        sim.reset()

        def env_ids(name: str) -> list[int]:
            paths = sim_utils.find_matching_prim_paths(f"/World/envs/env_.*/{name}")
            return sorted(int(p.split("/env_")[1].split("/")[0]) for p in paths)

        # sequential round-robin over two groups: group A -> envs (0, 2), group B -> envs (1, 3).
        assert env_ids("Robot_A") == [0, 2]
        assert env_ids("Robot_B") == [1, 3]
        # The static table must follow its group exactly -- present in every group-A env.
        assert env_ids("Table_A") == [0, 2]


def test_scene_publishes_plan_via_replicate(monkeypatch: pytest.MonkeyPatch):
    """A cfg-driven scene forwards the right plan and stage to cloner.replicate.

    Uses a test-seam fake to isolate this unit test from real backend dispatch; queue
    lifecycle is owned by :func:`replicate` itself (snapshot-and-clear) and does not
    need any cleanup hook here.
    """
    import isaaclab.cloner.replicate_session as replicate_session_module

    captured: list = []

    def fake_replicate(plan, *, stage):
        captured.append((plan, stage))

    monkeypatch.setattr(replicate_session_module, "replicate", fake_replicate)

    with build_simulation_context(device="cpu", auto_add_lighting=False, add_ground_plane=False) as sim:
        sim._app_control_on_stop_handle = None
        scene = InteractiveScene(MySceneCfg(num_envs=4, env_spacing=1.0))

    assert len(captured) == 1
    plan, stage = captured[0]
    assert plan.sources == ("/World/envs/env_0",)
    assert plan.destinations == ("/World/envs/env_{}",)
    assert plan.clone_mask.shape == (1, 4)
    assert stage is scene.stage


def test_collect_asset_cfgs_resolves_env_regex_macros(monkeypatch: pytest.MonkeyPatch):
    """_collect_asset_cfgs rewrites {ENV_REGEX_NS} macros and expands collections."""
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
    scene = _prepare_unit_scene(
        monkeypatch,
        2,
        objects=RigidObjectCollectionCfg(rigid_objects={"cube": cube_cfg, "shape": shape_cfg}),
    )

    cfgs = scene._collect_asset_cfgs()

    prim_paths = sorted(c.prim_path for c in cfgs)
    assert prim_paths == ["/World/envs/env_.*/Cube", "/World/envs/env_.*/Shape"]


def test_collect_asset_cfgs_orders_sensors_last(monkeypatch: pytest.MonkeyPatch):
    """Non-sensor cfgs precede sensor cfgs in _collect_asset_cfgs output."""
    from isaaclab.sensors import ContactSensorCfg

    sensor = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")
    body = SimpleNamespace(prim_path="{ENV_REGEX_NS}/Robot")
    scene = _prepare_unit_scene(monkeypatch, 1, sensor=sensor, body=body)

    cfgs = scene._collect_asset_cfgs()

    # Sensors come after non-sensor entities so they can bind to spawned bodies.
    assert cfgs.index(body) < cfgs.index(sensor)


def test_reset_filters_global_env_ids_to_selector_entity_indices(monkeypatch: pytest.MonkeyPatch):
    """Scene reset passes asset-local ids to selector-scoped assets and sensors."""
    scene = object.__new__(InteractiveScene)
    scene._selector = _make_selector(4)
    scene._selector.register("left", torch.tensor([0, 2]))
    scene._selector.register("right", torch.tensor([1, 3]))
    scene._selector.register_asset("robot", "left")
    scene._selector.register_asset("wrist_sensor", "right")
    monkeypatch.setattr(InteractiveScene, "device", property(lambda self: "cpu"))

    calls: dict[str, torch.Tensor | None] = {}

    def _recorder(name):
        return SimpleNamespace(reset=lambda env_ids=None: calls.setdefault(name, env_ids))

    scene._articulations = {"robot": _recorder("robot")}
    scene._deformable_objects = {}
    scene._rigid_objects = {"shared": _recorder("shared")}
    scene._surface_grippers = {}
    scene._rigid_object_collections = {}
    scene._sensors = {"wrist_sensor": _recorder("wrist_sensor")}

    scene.reset(torch.tensor([0, 1, 2, 3]))

    assert calls["robot"].tolist() == [0, 1]
    assert calls["wrist_sensor"].tolist() == [0, 1]
    assert calls["shared"].tolist() == [0, 1, 2, 3]


def test_collect_asset_cfgs_sets_valid_clone_combinations(monkeypatch: pytest.MonkeyPatch):
    """Scene collection builds a valid set from clone-combination config."""
    scene = _prepare_unit_scene(
        monkeypatch,
        4,
        object=SimpleNamespace(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.MultiAssetSpawnerCfg(
                assets_cfg=[sim_utils.ConeCfg(radius=0.1, height=0.2), sim_utils.SphereCfg(radius=0.1)]
            ),
        ),
        robot=SimpleNamespace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
        ),
    )
    scene.cloner_cfg = CloneCfg(
        clone_strategy=sequential,
        clone_combinations=[
            InclusionSet(assets=["object"], weight=1),
            InclusionSet(assets=["robot"], weight=1),
        ],
    )

    scene._collect_asset_cfgs()

    assert scene._clone_valid_set.tolist() == [[0, -1], [1, -1], [-1, 0]]


def test_clone_combinations_register_selector_asset_rows(monkeypatch: pytest.MonkeyPatch):
    """Selector asset rows are derived from the same valid combos used by ClonePlan."""
    scene = _prepare_unit_scene(
        monkeypatch,
        4,
        object=SimpleNamespace(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.MultiAssetSpawnerCfg(
                assets_cfg=[sim_utils.ConeCfg(radius=0.1, height=0.2), sim_utils.SphereCfg(radius=0.1)]
            ),
        ),
        robot=SimpleNamespace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
        ),
    )
    scene.cloner_cfg = CloneCfg(
        clone_strategy=sequential,
        clone_combinations=[
            InclusionSet(assets=["object"], weight=1),
            InclusionSet(assets=["robot"], weight=1),
        ],
    )

    plan = _build_unit_plan(scene)

    assert scene.selector["object"].env_ids.tolist() == [0, 1, 3]
    assert scene.selector["robot"].env_ids == slice(2, 3)
    assert scene.cfg.object.spawn.spawn_paths == ["/World/envs/env_0/Object", "/World/envs/env_1/Object"]
    assert scene.cfg.robot.spawn.spawn_path == "/World/envs/env_2/Robot"
    assert plan.sources == (
        "/World/envs/env_0/Object",
        "/World/envs/env_1/Object",
        "/World/envs/env_2/Robot",
    )
    assert plan.clone_mask.tolist() == [
        [True, False, False, True],
        [False, True, False, False],
        [False, False, True, False],
    ]


def test_clone_combinations_skip_assets_with_no_sampled_envs(monkeypatch: pytest.MonkeyPatch):
    """A zero-weight-only asset is omitted from spawning based on the planned mask."""
    scene = _prepare_unit_scene(
        monkeypatch,
        4,
        robot=SimpleNamespace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
        ),
        cabinet=SimpleNamespace(
            prim_path="{ENV_REGEX_NS}/Cabinet",
            spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
        ),
    )
    scene.cloner_cfg = CloneCfg(
        clone_strategy=sequential,
        clone_combinations=[
            InclusionSet(assets=["robot"], weight=1),
            InclusionSet(assets=["cabinet"], weight=0),
        ],
    )

    plan = _build_unit_plan(scene)

    assert scene.cfg.robot.spawn.spawn_path == "/World/envs/env_0/Robot"
    assert scene.cfg.cabinet.spawn.spawn_path is None
    assert scene._inactive_selector_assets == {"cabinet"}
    assert plan.clone_mask.tolist() == [
        [True, True, True, True],
        [False, False, False, False],
    ]


def test_add_entities_from_cfg_skips_inactive_selector_assets(monkeypatch: pytest.MonkeyPatch):
    """Entities with no sampled envs are not constructed."""
    scene = _prepare_unit_scene(
        monkeypatch,
        4,
        cabinet=SimpleNamespace(prim_path="{ENV_REGEX_NS}/Cabinet", spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))),
    )
    scene._inactive_selector_assets = {"cabinet"}
    scene._terrain = None
    scene._articulations = {}
    scene._deformable_objects = {}
    scene._rigid_objects = {}
    scene._rigid_object_collections = {}
    scene._sensors = {}
    scene._surface_grippers = {}
    scene._extras = {}

    scene._add_entities_from_cfg()

    assert "cabinet" not in scene.keys()


def test_make_clone_plan_sets_collection_member_paths(monkeypatch: pytest.MonkeyPatch):
    """Rigid object collection members are planned independently."""
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
    scene = _prepare_unit_scene(
        monkeypatch,
        4,
        objects=RigidObjectCollectionCfg(rigid_objects={"cube": cube_cfg, "shape": shape_cfg}),
    )

    plan = _build_unit_plan(scene)

    planned_cube = scene.cfg.objects.rigid_objects["cube"]
    planned_shape = scene.cfg.objects.rigid_objects["shape"]
    assert planned_cube.spawn.spawn_path == "/World/envs/env_0/Cube"
    assert planned_shape.spawn.spawn_paths == ["/World/envs/env_0/Shape", "/World/envs/env_1/Shape"]
    assert "/World/envs/env_{}/Cube" in plan.destinations
    assert "/World/envs/env_{}/Shape" in plan.destinations


def test_make_clone_plan_marks_unused_variants(monkeypatch: pytest.MonkeyPatch):
    """Unused variants keep a mask row but do not get spawned."""
    scene = _prepare_unit_scene(
        monkeypatch,
        2,
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

    plan = _build_unit_plan(scene)

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
                return  # found a difference -> success
    pytest.fail(f"No differing tensor found in nested state at {path}")
