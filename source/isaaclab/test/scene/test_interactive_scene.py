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
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.cloner import CloneCfg
from isaaclab.physics.scene_data_requirements import SceneDataRequirement
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import build_simulation_context
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

pytestmark = pytest.mark.integration


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


def test_scene_publishes_plan_via_replicate(monkeypatch: pytest.MonkeyPatch):
    """A cfg-driven scene forwards the right plan and stage to cloner.replicate.

    Uses a test-seam fake to isolate this unit test from real backend dispatch; queue
    lifecycle is owned by :func:`replicate` itself (snapshot-and-clear) and does not
    need any cleanup hook here.
    """
    import isaaclab.cloner.replicate_session as replicate_session_module

    captured: list = []

    def fake_replicate(plan, *, stage, replicate_physics=True):
        captured.append((plan, stage, replicate_physics))

    monkeypatch.setattr(replicate_session_module, "replicate", fake_replicate)

    with build_simulation_context(device="cpu", auto_add_lighting=False, add_ground_plane=False) as sim:
        sim._app_control_on_stop_handle = None
        scene = InteractiveScene(MySceneCfg(num_envs=4, env_spacing=1.0))

    assert len(captured) == 1
    plan, stage, replicate_physics = captured[0]
    assert plan.sources == ("/World/envs/env_0",)
    assert plan.destinations == ("/World/envs/env_{}",)
    assert plan.clone_mask.shape == (1, 4)
    assert stage is scene.stage
    assert replicate_physics is True


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("replicate_physics", [True, False])
def test_replicate_physics_flag_controls_physx_replicator(device, replicate_physics, setup_scene, monkeypatch):
    """replicate_physics=False must not register the PhysX replicator while envs still simulate.

    The True case asserts the spy actually intercepts registration, so the False case
    cannot pass vacuously.
    """
    physx_replicate_module = pytest.importorskip("isaaclab_physx.cloner.replicate")

    register_calls: list = []
    real_get_iface = physx_replicate_module.get_physx_replicator_interface

    class SpyInterface:
        def __init__(self, real):
            self._real = real

        def register_replicator(self, *args, **kwargs):
            register_calls.append(args)
            return self._real.register_replicator(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._real, name)

    monkeypatch.setattr(
        physx_replicate_module, "get_physx_replicator_interface", lambda: SpyInterface(real_get_iface())
    )

    make_scene, sim = setup_scene
    scene_cfg = make_scene(num_envs=3)
    scene_cfg.replicate_physics = replicate_physics
    scene = InteractiveScene(scene_cfg)
    if not scene.physics_backend.startswith("physx"):
        pytest.skip("PhysX replicator flag is only meaningful on a PhysX backend.")
    sim.reset()

    if replicate_physics:
        assert len(register_calls) > 0
    else:
        assert register_calls == []
    # all environments exist and simulate on both paths
    assert scene["rigid_obj"].data.root_pos_w.torch.shape[0] == 3
    assert scene["robot"].data.joint_pos.torch.shape[0] == 3
    for _ in range(2):
        sim.step()
        scene.update(sim.get_physics_dt())
    assert torch.isfinite(scene["rigid_obj"].data.root_pos_w.torch).all()
    assert torch.isfinite(scene["robot"].data.joint_pos.torch).all()


def test_cfg_cloning_contexts_override_backend_default(monkeypatch: pytest.MonkeyPatch):
    """AssetBaseCfg.cloning_contexts replaces the backend default stack for that asset."""
    import isaaclab.cloner.replicate_session as replicate_session_module
    from isaaclab.cloner import REPLICATION_QUEUE

    # keep the queue for inspection: the fake drain does not clear it
    monkeypatch.setattr(replicate_session_module, "replicate", lambda plan, *, stage, replicate_physics=True: None)

    with build_simulation_context(device="cpu", auto_add_lighting=False, add_ground_plane=False) as sim:
        sim._app_control_on_stop_handle = None
        scene_cfg = MySceneCfg(num_envs=2, env_spacing=1.0)
        scene_cfg.rigid_obj.cloning_contexts = ("isaaclab.cloner:UsdReplicateContext",)
        try:
            InteractiveScene(scene_cfg)
            queued_by_path = {cfg.prim_path: cfg for cfg in REPLICATION_QUEUE}
            # the override rides the queued cfg; resolution happens at replicate()
            assert queued_by_path["/World/envs/env_.*/RigidObj"].cloning_contexts == (
                "isaaclab.cloner:UsdReplicateContext",
            )
            # untouched asset resolves to the backend default stack at replicate()
            assert queued_by_path["/World/envs/env_.*/Robot"].cloning_contexts is None
        finally:
            REPLICATION_QUEUE.clear()


def test_collect_asset_cfgs_resolves_env_regex_macros():
    """_collect_asset_cfgs rewrites {ENV_REGEX_NS} macros and expands collections."""
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
        num_envs=2,
        objects=RigidObjectCollectionCfg(rigid_objects={"cube": cube_cfg, "shape": shape_cfg}),
    )
    scene.cloner_cfg = CloneCfg()
    scene._env_regex_ns = scene.cloner_cfg.clone_regex
    scene._env_ns = scene._env_regex_ns.rsplit("/", 1)[0]

    cfgs = scene._collect_asset_cfgs()

    prim_paths = sorted(c.prim_path for c in cfgs)
    assert prim_paths == ["/World/envs/env_.*/Cube", "/World/envs/env_.*/Shape"]


def test_collect_asset_cfgs_orders_sensors_last():
    """Non-sensor cfgs precede sensor cfgs in _collect_asset_cfgs output."""
    from isaaclab.sensors import ContactSensorCfg

    scene = object.__new__(InteractiveScene)
    sensor = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")
    body = SimpleNamespace(prim_path="{ENV_REGEX_NS}/Robot")
    scene.cfg = SimpleNamespace(num_envs=1, sensor=sensor, body=body)
    scene.cloner_cfg = CloneCfg()
    scene._env_ns = scene.cloner_cfg.clone_regex.rsplit("/", 1)[0]

    cfgs = scene._collect_asset_cfgs()

    # Sensors come after non-sensor entities so they can bind to spawned bodies.
    assert cfgs.index(body) < cfgs.index(sensor)


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
