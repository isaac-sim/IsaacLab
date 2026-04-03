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
import warp as wp

from pxr import PhysxSchema

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.scene import CollisionGroupCfg, InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import build_simulation_context
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


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
    scene["robot"].write_joint_state_to_sim(
        position=torch.rand_like(wp.to_torch(scene["robot"].data.joint_pos)),
        velocity=torch.rand_like(wp.to_torch(scene["robot"].data.joint_pos)),
    )
    next_state = scene.get_state(is_relative=False)
    assert_state_different(prev_state, next_state)
    scene.reset_to(prev_state, is_relative=False)
    assert_state_equal(prev_state, scene.get_state(is_relative=False))

    # test is relative == True
    prev_state = scene.get_state(is_relative=True)
    scene["robot"].write_joint_state_to_sim(
        position=torch.rand_like(wp.to_torch(scene["robot"].data.joint_pos)),
        velocity=torch.rand_like(wp.to_torch(scene["robot"].data.joint_pos)),
    )
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
    scene["robot"].write_joint_state_to_sim(
        position=torch.rand_like(wp.to_torch(scene["robot"].data.joint_pos)),
        velocity=torch.rand_like(wp.to_torch(scene["robot"].data.joint_pos)),
    )
    scene.reset_to(prev_state, env_ids=None)
    assert_state_equal(prev_state, scene.get_state())

    # test env_ids = torch tensor
    scene["robot"].write_joint_state_to_sim(
        position=torch.rand_like(wp.to_torch(scene["robot"].data.joint_pos)),
        velocity=torch.rand_like(wp.to_torch(scene["robot"].data.joint_pos)),
    )
    scene.reset_to(prev_state, env_ids=torch.arange(scene.num_envs, device=scene.device, dtype=torch.int32))
    assert_state_equal(prev_state, scene.get_state())


def test_clone_environments_non_cfg_invokes_visualizer_clone_fn(monkeypatch: pytest.MonkeyPatch):
    """Non-cfg clone path should execute visualizer clone callback with replicate args."""
    scene = object.__new__(InteractiveScene)
    scene.cfg = SimpleNamespace(replicate_physics=False, num_envs=3)
    scene.stage = object()
    scene.env_fmt = "/World/envs/env_{}"
    scene._ALL_INDICES = torch.arange(3, dtype=torch.long)
    scene._default_env_origins = torch.zeros((3, 3), dtype=torch.float32)
    scene._is_scene_setup_from_cfg = lambda: False

    # Avoid binding this unit test to global SimulationContext singleton state.
    monkeypatch.setattr(InteractiveScene, "device", property(lambda self: "cpu"))

    physics_calls = []
    visualizer_calls = []
    usd_calls = []

    def _physics_clone_fn(stage, *args, **kwargs):
        physics_calls.append((stage, args, kwargs))

    def _visualizer_clone_fn(stage, *args, **kwargs):
        visualizer_calls.append((stage, args, kwargs))

    def _usd_replicate(stage, *args, **kwargs):
        usd_calls.append((stage, args, kwargs))

    scene.cloner_cfg = SimpleNamespace(
        device="cpu",
        physics_clone_fn=_physics_clone_fn,
        visualizer_clone_fn=_visualizer_clone_fn,
    )
    monkeypatch.setattr("isaaclab.scene.interactive_scene.cloner.usd_replicate", _usd_replicate)

    scene.clone_environments(copy_from_source=False)
    assert len(physics_calls) == 1
    assert len(visualizer_calls) == 1
    assert len(usd_calls) == 1
    mapping = physics_calls[0][1][3]
    assert mapping.dtype == torch.bool
    assert mapping.shape == (1, scene.num_envs)

    physics_calls.clear()
    visualizer_calls.clear()
    usd_calls.clear()
    scene.clone_environments(copy_from_source=True)
    assert len(physics_calls) == 0
    assert len(visualizer_calls) == 1
    assert len(usd_calls) == 1


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


@configclass
class CollisionGroupSceneCfg(InteractiveSceneCfg):
    """Scene config for collision group tests."""

    cube_a = RigidObjectCfg(
        prim_path="/World/envs/env_.*/CubeA",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )
    cube_b = RigidObjectCfg(
        prim_path="/World/envs/env_.*/CubeB",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 0.5)),
    )
    cube_c = RigidObjectCfg(
        prim_path="/World/envs/env_.*/CubeC",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 0.5)),
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_collision_groups_prim_creation(device):
    """Verify collision group USD prims are created with correct structure."""
    with build_simulation_context(device=device, add_ground_plane=False) as sim:
        sim._app_control_on_stop_handle = None
        scene_cfg = CollisionGroupSceneCfg(num_envs=2, env_spacing=2.0)
        scene_cfg.collision_groups = {
            "group_ab": CollisionGroupCfg(assets=["cube_a", "cube_b"], collides_with=["group_c"]),
            "group_c": CollisionGroupCfg(assets=["cube_c"], collides_with=["group_ab"]),
        }
        scene = InteractiveScene(scene_cfg)

        # check scope prim exists
        scope_prim = scene.stage.GetPrimAtPath("/World/collisions")
        assert scope_prim.IsValid()

        # check each env/group combo
        for env_idx in range(2):
            for group_name in ["group_ab", "group_c"]:
                prim_path = f"/World/collisions/env{env_idx}_{group_name}"
                prim = scene.stage.GetPrimAtPath(prim_path)
                assert prim.IsValid(), f"Missing prim: {prim_path}"
                assert prim.GetPrimTypeInfo().GetTypeName() == "PhysicsCollisionGroup"

                # check expansion rule
                assert prim.GetAttribute("collection:colliders:expansionRule").Get() == "expandPrims"

                # check includes relationship has targets
                includes = prim.GetRelationship("collection:colliders:includes").GetTargets()
                assert len(includes) > 0

                # check filteredGroups relationship has targets
                filtered = prim.GetRelationship("physics:filteredGroups").GetTargets()
                assert len(filtered) > 0

        # check includes targets point to correct assets
        prim_ab_env0 = scene.stage.GetPrimAtPath("/World/collisions/env0_group_ab")
        includes_ab = [str(t) for t in prim_ab_env0.GetRelationship("collection:colliders:includes").GetTargets()]
        assert "/World/envs/env_0/CubeA" in includes_ab
        assert "/World/envs/env_0/CubeB" in includes_ab

        prim_c_env0 = scene.stage.GetPrimAtPath("/World/collisions/env0_group_c")
        includes_c = [str(t) for t in prim_c_env0.GetRelationship("collection:colliders:includes").GetTargets()]
        assert "/World/envs/env_0/CubeC" in includes_c

        # check InvertCollisionGroupFilterAttr is set
        physx_scene = PhysxSchema.PhysxSceneAPI(scene.stage.GetPrimAtPath(scene.physics_scene_path))
        assert physx_scene.GetInvertCollisionGroupFilterAttr().Get() is True


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_collision_groups_mutual_agreement(device):
    """Verify that collisions require mutual agreement between groups.

    group_a wants to collide with group_b, but group_b says collides_with=[].
    Since both sides must agree, they should NOT collide.
    """
    with build_simulation_context(device=device, add_ground_plane=False) as sim:
        sim._app_control_on_stop_handle = None
        scene_cfg = CollisionGroupSceneCfg(num_envs=1, env_spacing=2.0)
        scene_cfg.collision_groups = {
            "group_a": CollisionGroupCfg(assets=["cube_a"], collides_with=["group_b"]),
            "group_b": CollisionGroupCfg(assets=["cube_b"], collides_with=[]),
        }
        scene = InteractiveScene(scene_cfg)

        # group_a should only have self (group_b rejected)
        prim_a = scene.stage.GetPrimAtPath("/World/collisions/env0_group_a")
        filtered_a = [str(t) for t in prim_a.GetRelationship("physics:filteredGroups").GetTargets()]
        assert "/World/collisions/env0_group_a" in filtered_a
        assert "/World/collisions/env0_group_b" not in filtered_a

        # group_b should only have self
        prim_b = scene.stage.GetPrimAtPath("/World/collisions/env0_group_b")
        filtered_b = [str(t) for t in prim_b.GetRelationship("physics:filteredGroups").GetTargets()]
        assert "/World/collisions/env0_group_b" in filtered_b
        assert "/World/collisions/env0_group_a" not in filtered_b


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_collision_groups_collides_with_none(device):
    """Verify collides_with=None means willing to collide with all, but requires mutual agreement."""
    with build_simulation_context(device=device, add_ground_plane=False) as sim:
        sim._app_control_on_stop_handle = None
        scene_cfg = CollisionGroupSceneCfg(num_envs=1, env_spacing=2.0)
        scene_cfg.collision_groups = {
            "group_a": CollisionGroupCfg(assets=["cube_a"], collides_with=None),  # willing to collide with all
            "group_b": CollisionGroupCfg(assets=["cube_b"], collides_with=[]),  # isolated
            "group_c": CollisionGroupCfg(assets=["cube_c"], collides_with=None),  # willing to collide with all
        }
        scene = InteractiveScene(scene_cfg)

        # group_a (None) + group_c (None) → both agree → collide
        prim_a = scene.stage.GetPrimAtPath("/World/collisions/env0_group_a")
        filtered_a = [str(t) for t in prim_a.GetRelationship("physics:filteredGroups").GetTargets()]
        assert "/World/collisions/env0_group_a" in filtered_a
        assert "/World/collisions/env0_group_c" in filtered_a

        # group_a (None) + group_b ([]) → group_b rejects → no collision
        assert "/World/collisions/env0_group_b" not in filtered_a


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_collision_groups_invalid_asset_name(device):
    """Verify ValueError when collision group references nonexistent asset."""
    with build_simulation_context(device=device, add_ground_plane=False) as sim:
        sim._app_control_on_stop_handle = None
        scene_cfg = CollisionGroupSceneCfg(num_envs=1, env_spacing=2.0)
        scene_cfg.collision_groups = {
            "group_a": CollisionGroupCfg(assets=["nonexistent_asset"], collides_with=[]),
        }
        with pytest.raises(ValueError, match="nonexistent_asset"):
            InteractiveScene(scene_cfg)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_collision_groups_invalid_group_reference(device):
    """Verify ValueError when collides_with references undefined group."""
    with build_simulation_context(device=device, add_ground_plane=False) as sim:
        sim._app_control_on_stop_handle = None
        scene_cfg = CollisionGroupSceneCfg(num_envs=1, env_spacing=2.0)
        scene_cfg.collision_groups = {
            "group_a": CollisionGroupCfg(assets=["cube_a"], collides_with=["nonexistent_group"]),
        }
        with pytest.raises(ValueError, match="nonexistent_group"):
            InteractiveScene(scene_cfg)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_collision_groups_none_preserves_existing_behavior(device):
    """Verify that collision_groups=None (default) doesn't create per-env collision group prims."""
    with build_simulation_context(device=device, add_ground_plane=False) as sim:
        sim._app_control_on_stop_handle = None
        scene_cfg = CollisionGroupSceneCfg(num_envs=2, env_spacing=2.0)
        # collision_groups defaults to None
        scene = InteractiveScene(scene_cfg)

        # no per-env collision group prims should exist
        env0_group_prim = scene.stage.GetPrimAtPath("/World/collisions/env0_group_ab")
        assert not env0_group_prim.IsValid()


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
