# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, Asset, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.cloner import CloneCfg
from isaaclab.markers import SPHERE_MARKER_CFG, VisualizationMarkers
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import build_simulation_context
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

pytestmark = pytest.mark.integration


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration."""

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/SimpleArticulation/revolute_articulation.usd",
        ),
        actuators={"joint": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=100.0, damping=1.0)},
    )
    rigid_obj = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RigidObj",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            rigid_props=PhysxRigidBodyCfg(disable_gravity=False),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
        ),
    )


@configclass
class DeferredMarkerAssetCfg(AssetBaseCfg):
    """Authoring-only asset whose optional visualization starts disabled."""

    visualizer_cfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Deferred")


@configclass
class GlobalAssetsSceneCfg(InteractiveSceneCfg):
    """Scene made of authoring-only assets and markers that live at global prim roots."""

    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DistantLightCfg())
    goal = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Goal")
    prop = DeferredMarkerAssetCfg(prim_path="/World/Prop", spawn=sim_utils.DistantLightCfg(), debug_vis=False)


@pytest.fixture
def sim(request):
    """Simulation context on the requested device with ground plane and lighting."""
    device = getattr(request, "param", "cpu")
    with build_simulation_context(device=device, auto_add_lighting=True, add_ground_plane=True) as sim:
        sim._app_control_on_stop_handle = None
        yield sim


def _flatten(state: dict, prefix: str = "") -> dict[str, torch.Tensor]:
    """Flatten a nested scene state into ``{"group.entity.field": tensor}``."""
    flat = {}
    for key, value in state.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def _assert_state_equal(s1: dict, s2: dict) -> None:
    f1, f2 = _flatten(s1), _flatten(s2)
    assert f1.keys() == f2.keys()
    for key in f1:
        assert torch.equal(f1[key], f2[key]), f"state mismatch at {key}"


def _assert_state_different(s1: dict, s2: dict) -> None:
    f1, f2 = _flatten(s1), _flatten(s2)
    assert f1.keys() == f2.keys()
    assert any(not torch.equal(f1[key], f2[key]) for key in f1), "no differing state found"


def _perturb_robot(scene: InteractiveScene) -> None:
    joint_pos = scene["robot"].data.joint_pos.torch
    scene["robot"].write_joint_position_to_sim_index(position=torch.rand_like(joint_pos))
    scene["robot"].write_joint_velocity_to_sim_index(velocity=torch.rand_like(joint_pos))


@pytest.mark.parametrize("sim", ["cuda:0", "cpu"], indirect=True)
def test_get_state_reset_to_round_trip(sim):
    """``reset_to`` restores a captured state in both frames and for every ``env_ids`` input type."""
    scene = InteractiveScene(MySceneCfg(num_envs=4, env_spacing=1.0))
    sim.reset()
    _assert_state_different(scene.get_state(is_relative=False), scene.get_state(is_relative=True))

    for is_relative in (False, True):
        prev_state = scene.get_state(is_relative=is_relative)
        _perturb_robot(scene)
        _assert_state_different(prev_state, scene.get_state(is_relative=is_relative))
        scene.reset_to(prev_state, is_relative=is_relative)
        _assert_state_equal(prev_state, scene.get_state(is_relative=is_relative))

    prev_state = scene.get_state()
    for env_ids in (None, torch.arange(scene.num_envs, device=scene.device, dtype=torch.int32)):
        _perturb_robot(scene)
        scene.reset_to(prev_state, env_ids=env_ids)
        _assert_state_equal(prev_state, scene.get_state())


def test_relative_deformable_state():
    """Deformable nodal positions shift by the per-env origin in relative states, on read and on reset."""
    env_origins = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    nodal_position = torch.arange(60, dtype=torch.float32).reshape(4, 5, 3)
    written_state = {}
    deformable = SimpleNamespace(
        data=SimpleNamespace(
            nodal_pos_w=SimpleNamespace(torch=nodal_position),
            nodal_vel_w=SimpleNamespace(torch=torch.zeros_like(nodal_position)),
        ),
        write_nodal_pos_to_sim=lambda value, env_ids: written_state.update(position=value, env_ids=env_ids),
        write_nodal_velocity_to_sim=lambda value, env_ids: written_state.update(velocity=value),
    )
    scene = SimpleNamespace(
        device="cpu",
        env_origins=env_origins,
        _articulations={},
        _cable_objects={},
        _deformable_objects={"object": deformable},
        _rigid_objects={},
        _surface_grippers={},
        _rigid_object_collections={},
        write_data_to_sim=lambda: None,
    )

    state = InteractiveScene.get_state(scene, is_relative=True)
    torch.testing.assert_close(
        state["deformable_object"]["object"]["nodal_position"], nodal_position - env_origins[:, None, :]
    )

    env_ids = torch.tensor([3, 1])
    reset_nodal_position = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3)
    reset_nodal_velocity = torch.ones_like(reset_nodal_position)
    reset_state = {
        "deformable_object": {
            "object": {"nodal_position": reset_nodal_position, "nodal_velocity": reset_nodal_velocity}
        }
    }
    InteractiveScene.reset_to(scene, reset_state, env_ids=env_ids, is_relative=True)

    torch.testing.assert_close(written_state["position"], reset_nodal_position + env_origins[env_ids, None, :])
    torch.testing.assert_close(written_state["velocity"], reset_nodal_velocity)
    torch.testing.assert_close(written_state["env_ids"], env_ids)


def test_scene_publishes_plan_before_replicate(monkeypatch: pytest.MonkeyPatch):
    """A cfg-driven scene publishes the exact plan it forwards to replication."""
    import isaaclab.cloner.replicate_session as replicate_session_module

    captured: list = []

    def fake_replicate(plan, *, replicate_physics=True):
        captured.append((plan, replicate_physics, sim_utils.SimulationContext.instance().get_clone_plan()))

    monkeypatch.setattr(replicate_session_module, "replicate", fake_replicate)

    with build_simulation_context(device="cpu", auto_add_lighting=False, add_ground_plane=False) as sim:
        sim._app_control_on_stop_handle = None
        InteractiveScene(MySceneCfg(num_envs=4, env_spacing=1.0))

    assert len(captured) == 1
    plan, replicate_physics, published = captured[0]
    assert published is plan
    assert plan.sources == ("/World/envs/env_0",)
    assert plan.destinations == ("/World/envs/env_{}",)
    assert plan.clone_mask.shape == (1, 4)
    assert replicate_physics is True


def test_scene_constructs_global_assets_and_markers():
    """Authoring-only assets and markers (even on disabled debug owners) are built and own plan global roots."""
    with build_simulation_context(device="cpu", auto_add_lighting=False, add_ground_plane=False) as sim:
        scene = InteractiveScene(GlobalAssetsSceneCfg(num_envs=1, env_spacing=1.0))

        assert isinstance(scene["light"], Asset)
        assert scene["light"].cfg.prim_path == "/World/Light"
        assert scene["light"].prim == scene.stage.GetPrimAtPath("/World/Light")
        assert isinstance(scene["goal"], VisualizationMarkers)
        expected_roots = ("/World/Light", "/Visuals/Goal", "/World/Prop", "/Visuals/Deferred")
        assert sim.get_clone_plan().global_paths == expected_roots
        assert scene.keys() == ["terrain", "light", "goal", "prop"]
        with pytest.raises(KeyError, match="Available Entities"):
            scene["missing"]


def test_empty_scene_leaves_clone_lifecycle_to_caller():
    """An empty scene authors one prototype and leaves its replication to the direct task."""
    with build_simulation_context(device="cpu", auto_add_lighting=False, add_ground_plane=False) as sim:
        sim._app_control_on_stop_handle = None
        scene = InteractiveScene(InteractiveSceneCfg(num_envs=4, env_spacing=1.0))

        assert sim.get_clone_plan() is None
        env_template = scene.cfg.clone_cfg.clone_template
        grid_positions = cloner.grid_transforms(4, 1.0)[0]
        torch.testing.assert_close(scene.env_origins, torch.from_numpy(grid_positions))
        env_0 = scene.stage.GetPrimAtPath(env_template.format(0))
        np.testing.assert_allclose(sim_utils.resolve_prim_pose(env_0)[0], grid_positions[0])
        assert all(not scene.stage.GetPrimAtPath(env_template.format(i)).IsValid() for i in range(1, 4))

        cube_cfg = RigidObjectCfg(
            prim_path=f"{env_template.format('[^/]+')}/Cube",
            spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
            cloning_contexts=(cloner.UsdReplicateContext,),
        )
        positions = grid_positions + np.asarray((0.25, 0.5, 0.75), dtype=np.float32)
        plan = cloner.clone_plan_from_env_0(scene.cfg.clone_cfg, (cube_cfg,), 4, 1.0, positions=positions)
        cube_cfg.class_type(cube_cfg)
        cloner.replicate(plan)

        assert sim.get_clone_plan() is plan
        assert all(scene.stage.GetPrimAtPath(f"{env_template.format(i)}/Cube").IsValid() for i in range(4))
        torch.testing.assert_close(scene.env_origins, torch.from_numpy(positions))


@pytest.mark.parametrize("sim", ["cuda:0"], indirect=True)
@pytest.mark.parametrize("replicate_physics", [True, False])
def test_replicate_physics_flag_controls_physx_replicator(sim, replicate_physics, monkeypatch):
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

    scene = InteractiveScene(MySceneCfg(num_envs=3, env_spacing=1.0, replicate_physics=replicate_physics))
    if not scene.physics_backend.startswith("physx"):
        pytest.skip("PhysX replicator flag is only meaningful on a PhysX backend.")
    sim.reset()

    assert bool(register_calls) is replicate_physics
    # all environments exist and simulate on both paths
    assert scene["rigid_obj"].data.root_pos_w.torch.shape[0] == 3
    assert scene["robot"].data.joint_pos.torch.shape[0] == 3
    for _ in range(2):
        sim.step()
        scene.update(sim.get_physics_dt())
    assert torch.isfinite(scene["rigid_obj"].data.root_pos_w.torch).all()
    assert torch.isfinite(scene["robot"].data.joint_pos.torch).all()


def _bare_scene(**cfg_entities) -> InteractiveScene:
    """Build an uninitialized scene that only carries the cfg fields ``_collect_asset_cfgs`` reads."""
    scene = object.__new__(InteractiveScene)
    scene.cfg = SimpleNamespace(num_envs=2, **cfg_entities)
    scene.cloner_cfg = CloneCfg()
    scene._env_fmt = scene.cloner_cfg.clone_template
    return scene


def test_collect_asset_cfgs_resolves_env_regex_macros_and_declares_globals():
    """The composition root separates cloneable configs from shared prim roots."""
    cube_cfg = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Cube", spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)))
    shape_cfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shape",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[sim_utils.ConeCfg(radius=0.1, height=0.2), sim_utils.SphereCfg(radius=0.1)]
        ),
    )
    scene = _bare_scene(
        objects=RigidObjectCollectionCfg(rigid_objects={"cube": cube_cfg, "shape": shape_cfg}),
        ground=AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg()),
    )

    cfgs, global_paths, _ = scene._collect_asset_cfgs()

    assert sorted(c.prim_path for c in cfgs) == ["/World/envs/env_[^/]+/Cube", "/World/envs/env_[^/]+/Shape"]
    assert global_paths == ("/World/Ground",)


def test_collect_asset_cfgs_excludes_entities_without_spawners():
    """Sensors without spawners add no clone rows but still declare their debug-marker roots."""
    sensor = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")
    scene = _bare_scene(sensor=sensor)

    cfgs, global_paths, _ = scene._collect_asset_cfgs()

    assert cfgs == []
    assert global_paths == (sensor.visualizer_cfg.prim_path,)
