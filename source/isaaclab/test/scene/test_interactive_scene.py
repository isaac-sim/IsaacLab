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
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.physics.scene_data_requirements import SceneDataRequirement
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
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
        position=torch.rand_like(scene["robot"].data.joint_pos.torch),
        velocity=torch.rand_like(scene["robot"].data.joint_pos.torch),
    )
    next_state = scene.get_state(is_relative=False)
    assert_state_different(prev_state, next_state)
    scene.reset_to(prev_state, is_relative=False)
    assert_state_equal(prev_state, scene.get_state(is_relative=False))

    # test is relative == True
    prev_state = scene.get_state(is_relative=True)
    scene["robot"].write_joint_state_to_sim(
        position=torch.rand_like(scene["robot"].data.joint_pos.torch),
        velocity=torch.rand_like(scene["robot"].data.joint_pos.torch),
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
        position=torch.rand_like(scene["robot"].data.joint_pos.torch),
        velocity=torch.rand_like(scene["robot"].data.joint_pos.torch),
    )
    scene.reset_to(prev_state, env_ids=None)
    assert_state_equal(prev_state, scene.get_state())

    # test env_ids = torch tensor
    scene["robot"].write_joint_state_to_sim(
        position=torch.rand_like(scene["robot"].data.joint_pos.torch),
        velocity=torch.rand_like(scene["robot"].data.joint_pos.torch),
    )
    scene.reset_to(prev_state, env_ids=torch.arange(scene.num_envs, device=scene.device, dtype=torch.int32))
    assert_state_equal(prev_state, scene.get_state())


def test_clone_environments_non_cfg_publishes_clone_plans(monkeypatch: pytest.MonkeyPatch):
    """Non-cfg clone path must dispatch physics + USD replicate and publish a ``ClonePlan``.

    Replaces the old test that asserted a per-call visualizer clone callback was invoked. The
    visualizer-fn callback was removed in favor of providers reading
    :meth:`SimulationContext.get_clone_plans`; this test asserts the new contract: even
    without prototype templates, the scene synthesizes a single trivial ClonePlan.
    """
    from isaaclab.cloner import ClonePlan

    scene = object.__new__(InteractiveScene)
    scene.cfg = SimpleNamespace(replicate_physics=False, num_envs=3)
    scene.stage = object()
    scene.physics_backend = "physx"
    scene._sensors = {}

    set_plans_calls: list = []
    sim_state: dict = {"plans": {}}

    def _set_clone_plans(plans):
        sim_state["plans"] = plans
        set_plans_calls.append(plans)

    scene.sim = SimpleNamespace(
        get_scene_data_requirements=lambda: SceneDataRequirement(),
        update_scene_data_requirements=lambda requirements: None,
        set_clone_plans=_set_clone_plans,
        get_clone_plans=lambda: sim_state["plans"],
    )
    scene.env_fmt = "/World/envs/env_{}"
    scene._ALL_INDICES = torch.arange(3, dtype=torch.long)
    scene._default_env_origins = torch.zeros((3, 3), dtype=torch.float32)
    scene._is_scene_setup_from_cfg = lambda: False

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
    # Plans are published once per clone, regardless of physics/usd flag combinations.
    assert len(set_plans_calls) == 1
    plans = set_plans_calls[-1]
    assert set(plans.keys()) == {scene.env_fmt}
    plan = plans[scene.env_fmt]
    assert isinstance(plan, ClonePlan)
    assert plan.dest_template == scene.env_fmt
    assert plan.prototype_paths == [scene.env_fmt.format(0)]
    assert plan.clone_mask.shape == (1, scene.num_envs)
    assert scene.clone_plans is plans

    physics_calls.clear()
    usd_calls.clear()
    set_plans_calls.clear()
    scene.clone_environments(copy_from_source=True)
    assert len(physics_calls) == 0
    assert len(usd_calls) == 1
    assert len(set_plans_calls) == 1


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
