# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for PhysX-dependent cloner utilities."""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher
from isaaclab.assets import AssetBaseCfg
from isaaclab.test.utils import DeviceScope, test_devices

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_physx.cloner import PhysxReplicateContext, physx_replicate
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.cloner import (
    UsdReplicateContext,
    _fabric_notices,
    disabled_fabric_change_notifies,
    make_clone_plan,
    usd_replicate,
)
from isaaclab.sim import build_simulation_context


def _make_flat_clone_plan(num_variants: int, num_clones: int, destination: str):
    """Raw paths and a round-robin mask for testing the native replication API."""
    chosen = np.arange(num_clones) % num_variants
    mask = np.zeros((num_variants, num_clones), dtype=np.bool_)
    mask[chosen, np.arange(num_clones)] = True
    sources = tuple(destination.format(i) for i in range(num_variants))
    destinations = tuple([destination] * num_variants)
    return sources, destinations, mask


wp.init()

pytestmark = pytest.mark.isaacsim_ci


@pytest.fixture(params=["cpu", "cuda"])
def sim(request):
    """Provide a fresh simulation context for each test on CPU and CUDA."""
    with build_simulation_context(device=request.param, dt=0.01, add_lighting=False) as sim:
        yield sim


# Replicator bookkeeping (mapping checks, world lists, Fabric notices) does not depend on the
# simulation device, so those tests run on CUDA only.
cuda_only = pytest.mark.parametrize("sim", test_devices(DeviceScope.DEFAULT_CUDA), indirect=True)


@cuda_only
def test_physx_replicate_validates_mapping_shape(sim):
    """Mapping rows and columns match the declared sources and environments."""
    env_ids = np.arange(2, dtype=np.int64)
    with pytest.raises(ValueError, match="mapping must have shape"):
        physx_replicate(
            sim_utils.get_current_stage(),
            sources=["/World/template/A"],
            destinations=["/World/envs/env_{}/A"],
            env_ids=env_ids,
            mapping=np.ones((1, len(env_ids) + 1), dtype=np.bool_),
        )


def _make_mock_physx_rep():
    """Return (mock_rep, replicate_calls) where replicate_calls accumulates num_worlds per call.

    ``mock_rep.register_replicator`` immediately invokes attach_fn + attach_end_fn so the callbacks
    fire synchronously inside ``physx_replicate``, making the calls observable in tests.
    """
    from unittest.mock import MagicMock

    replicate_calls: list[int] = []
    mock_rep = MagicMock()
    mock_rep.replicate.side_effect = lambda _sid, _src, num_worlds, **kw: replicate_calls.append(num_worlds)

    def _fake_register(_stage_id, attach_fn, attach_end_fn, rename_fn):
        attach_fn(_stage_id)
        attach_end_fn(_stage_id)

    mock_rep.register_replicator.side_effect = _fake_register
    return mock_rep, replicate_calls


def _make_mock_physx_rep_detailed():
    """Return (mock_rep, replicate_calls, attach_excluded) for fine-grained inspection.

    ``replicate_calls`` is a list of ``(src, num_worlds)`` tuples — one entry per
    ``rep.replicate`` invocation, preserving the source path for heterogeneous checks.
    ``attach_excluded`` is the list of paths returned by ``attach_fn`` (i.e. the paths
    that the replicator will exclude from its USD stage parse).
    """
    from unittest.mock import MagicMock

    replicate_calls: list[tuple[str, int]] = []
    attach_excluded: list[str] = []
    mock_rep = MagicMock()
    mock_rep.replicate.side_effect = lambda _sid, src, num_worlds, **kw: replicate_calls.append((src, num_worlds))

    def _fake_register(_stage_id, attach_fn, attach_end_fn, rename_fn):
        excluded = attach_fn(_stage_id) or []
        attach_excluded.extend(excluded)
        attach_end_fn(_stage_id)

    mock_rep.register_replicator.side_effect = _fake_register
    return mock_rep, replicate_calls, attach_excluded


@cuda_only
def test_physx_replicate_context_consumes_plan(sim):
    """PhysxReplicateContext reads its mapping from the shared clone plan."""
    from unittest.mock import patch

    stage = sim_utils.get_current_stage()
    sim_utils.create_prim("/World/envs", "Xform")
    for i in range(3):
        sim_utils.create_prim(f"/World/envs/env_{i}", "Xform")

    mock_rep, replicate_calls = _make_mock_physx_rep()
    with patch("isaaclab_physx.cloner.replicate.get_physx_replicator_interface", return_value=mock_rep):
        ctx = PhysxReplicateContext(stage)
        plan = make_clone_plan((AssetBaseCfg(prim_path="/World/envs/env_[^/]+/Object"),), ((0,),), 3)
        sim.clone_contexts[UsdReplicateContext] = UsdReplicateContext(stage, plan)
        ctx.replicate(plan, (0,))

    assert replicate_calls == [2]


@cuda_only
@pytest.mark.parametrize(
    "num_envs,src,expected_worlds",
    [
        (3, "/World/envs/env_0", [2]),
        (1, "/World/envs/env_0", []),
        (3, "/World/template/Robot", [3]),
    ],
)
def test_physx_replicate_world_counts(sim, num_envs, src, expected_worlds):
    """physx_replicate calls rep.replicate with the correct world count (exclude-self).

    With ``exclude_self_replication=True`` (default), the source environment is excluded
    from the replication targets when it also maps to other environments.  A source at
    ``env_0`` mapping to ``[0, 1, 2]`` only replicates to ``[1, 2]`` (2 worlds).
    With ``num_envs == 1`` the ``num_envs > 1`` guard skips registration entirely.
    Non-env sources (e.g. ``/World/template/Robot``) are never excluded because the
    self-id is not a digit.
    """
    from unittest.mock import patch

    stage = sim_utils.get_current_stage()
    sim_utils.create_prim("/World/envs", "Xform")
    sim_utils.create_prim("/World/template", "Xform")
    sim_utils.create_prim("/World/template/Robot", "Xform")
    for i in range(num_envs):
        sim_utils.create_prim(f"/World/envs/env_{i}", "Xform")

    mock_rep, replicate_calls = _make_mock_physx_rep()
    with patch("isaaclab_physx.cloner.replicate.get_physx_replicator_interface", return_value=mock_rep):
        physx_replicate(
            stage,
            sources=[src],
            destinations=["/World/envs/env_{}"],
            env_ids=np.arange(num_envs, dtype=np.int64),
            mapping=np.ones((1, num_envs), dtype=np.bool_),
        )

    assert replicate_calls == expected_worlds, (
        f"Expected replicate world counts {expected_worlds}, got {replicate_calls}"
    )


def test_physx_replicate_isolated_source_loaded_without_replication(sim):
    """A single-env source (worlds=[self]) is correctly loaded after physx_replicate.

    When there is only one environment and the source maps to itself,
    ``exclude_self_replication=True`` (default) causes physx_replicate to skip
    replication entirely. The prim already exists from USD, so after ``sim.reset()``
    PhysX must still be able to find the rigid body at the env path.
    """
    stage = sim_utils.get_current_stage()

    sim_utils.create_prim("/World/envs", "Xform")
    sim_utils.create_prim("/World/template", "Xform")
    sphere_cfg = sim_utils.SphereCfg(
        radius=0.1,
        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
        mass_props=sim_utils.MassCfg(mass=1.0),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
    )
    sphere_cfg.func("/World/envs/env_0/Sphere", sphere_cfg)

    physx_replicate(
        stage,
        sources=["/World/envs/env_0/Sphere"],
        destinations=["/World/envs/env_{}/Sphere"],
        env_ids=np.array([0], dtype=np.int64),
        mapping=np.ones((1, 1), dtype=np.bool_),
    )

    sim.reset()

    physics_sim_view = sim.physics_manager.get_physics_sim_view()
    physx_view = physics_sim_view.create_rigid_body_view("/World/envs/env_*/Sphere")
    assert physx_view is not None and physx_view.count == 1, (
        f"Expected 1 rigid body at /World/envs/env_0/Sphere, got {'None' if physx_view is None else physx_view.count}."
    )


@cuda_only
def test_physx_replicate_heterogeneous_isolated_sources(sim):
    """physx_replicate handles heterogeneous sources excluding self from world lists.

    This is the Lift scenario: multiple object types, each with a designated proto-env.
    With ``exclude_self_replication=True`` (default), self is removed from the world list
    only when the source also maps to other environments.  Self-only sources keep self so
    that ``rep.replicate()`` still fires and the source prim gets its physics body (since
    ``/World/envs`` is excluded from normal PhysX parsing).

    Sources and expected behaviour:
      env_0/Object → worlds [0, 2, 4]   → exclude 0 → replicate to [2, 4]  (2 worlds)
      env_5/Object → worlds [5]          → exclude 5 → keep [5]             (1 world)
      env_7/Object → worlds [7, 11]      → exclude 7 → replicate to [11]   (1 world)
    """
    from unittest.mock import patch

    num_envs = 16
    stage = sim_utils.get_current_stage()
    sim_utils.create_prim("/World/template", "Xform")
    for i in range(num_envs):
        sim_utils.create_prim(f"/World/envs/env_{i}", "Xform")

    mapping = np.zeros((3, num_envs), dtype=np.bool_)
    mapping[0, [0, 2, 4]] = True
    mapping[1, [5]] = True
    mapping[2, [7, 11]] = True

    mock_rep, replicate_calls, attach_excluded = _make_mock_physx_rep_detailed()
    with patch("isaaclab_physx.cloner.replicate.get_physx_replicator_interface", return_value=mock_rep):
        physx_replicate(
            stage,
            sources=["/World/envs/env_0/Object", "/World/envs/env_5/Object", "/World/envs/env_7/Object"],
            destinations=["/World/envs/env_{}/Object"] * 3,
            env_ids=np.arange(num_envs, dtype=np.int64),
            mapping=mapping,
        )

    expected = [
        ("/World/envs/env_0/Object", 2),
        ("/World/envs/env_5/Object", 1),
        ("/World/envs/env_7/Object", 1),
    ]
    assert replicate_calls == expected, f"Expected {expected}, got {replicate_calls}."

    # attach_fn always returns ["/World/template", "/World/envs"] so the replicator
    # owns all env prims.  Self-only sources get their physics body from the
    # rep.replicate() call itself.
    assert "/World/template" in attach_excluded
    assert "/World/envs" in attach_excluded


def test_direct_clone_plan_multi_asset(sim):
    """Clone representative env sources directly and exercise both USD and PhysX."""
    num_clones = 32
    sim_utils.create_prim("/World/envs", "Xform")
    for i in range(num_clones):
        sim_utils.create_prim(f"/World/envs/env_{i}", "Xform", translation=(0, 0, 0))

    cfg = sim_utils.MultiAssetSpawnerCfg(
        assets_cfg=[
            sim_utils.ConeCfg(
                radius=0.3,
                height=0.6,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                mass_props=sim_utils.MassCfg(mass=100.0),
            ),
            sim_utils.CuboidCfg(
                size=(0.3, 0.3, 0.3),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
            ),
            sim_utils.SphereCfg(
                radius=0.3,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
            ),
        ],
        rigid_props=PhysxRigidBodyCfg(solver_position_iteration_count=4, solver_velocity_iteration_count=0),
        mass_props=sim_utils.MassCfg(mass=1.0),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
    )
    sources, destinations, clone_mask = _make_flat_clone_plan(
        num_variants=len(cfg.assets_cfg),
        num_clones=num_clones,
        destination="/World/envs/env_{}/Object",
    )
    cfg.spawn_paths = list(sources)
    prim = cfg.func("/World/unused", cfg)
    assert prim.IsValid()

    stage = sim_utils.get_current_stage()
    env_ids = np.arange(num_clones, dtype=np.int64)
    physx_replicate(stage, sources, destinations, env_ids, clone_mask)
    usd_replicate(stage, sources, destinations, env_ids, clone_mask)

    primitive_prims = sim_utils.get_all_matching_child_prims(
        "/World/envs", predicate=lambda prim: prim.GetTypeName() in ["Cone", "Cube", "Sphere"]
    )

    for i, primitive_prim in enumerate(primitive_prims):
        modulus = i % 3
        if modulus == 0:
            assert primitive_prim.GetTypeName() == "Cone"
        elif modulus == 1:
            assert primitive_prim.GetTypeName() == "Cube"
        else:
            assert primitive_prim.GetTypeName() == "Sphere"

    sim.reset()
    physics_sim_view = sim.physics_manager.get_physics_sim_view()
    physx_view = physics_sim_view.create_rigid_body_view("/World/envs/env_*/Object")
    assert physx_view.count == num_clones


def _run_colocation_collision_filter(sim, asset_cfg, expected_types, assert_count=False):
    """Shared harness for colocated collision filter checks across devices."""
    num_clones = 32
    sim_utils.create_prim("/World/envs", "Xform")
    for i in range(num_clones):
        sim_utils.create_prim(f"/World/envs/env_{i}", "Xform", translation=(0, 0, 0))

    num_variants = len(asset_cfg.assets_cfg) if isinstance(asset_cfg, sim_utils.MultiAssetSpawnerCfg) else 1
    sources, destinations, clone_mask = _make_flat_clone_plan(
        num_variants=num_variants,
        num_clones=num_clones,
        destination="/World/envs/env_{}/Object",
    )
    if isinstance(asset_cfg, sim_utils.MultiAssetSpawnerCfg):
        asset_cfg.spawn_paths = list(sources)
        prim = asset_cfg.func("/World/unused", asset_cfg)
    else:
        prim = asset_cfg.func(sources[0], asset_cfg)
    assert prim.IsValid()

    stage = sim_utils.get_current_stage()
    env_ids = np.arange(num_clones, dtype=np.int64)
    physx_replicate(stage, sources, destinations, env_ids, clone_mask)
    usd_replicate(stage, sources, destinations, env_ids, clone_mask)

    primitive_prims = sim_utils.get_all_matching_child_prims(
        "/World/envs", predicate=lambda prim: prim.GetTypeName() in expected_types
    )

    if assert_count:
        assert len(primitive_prims) == num_clones

    for i, primitive_prim in enumerate(primitive_prims):
        assert primitive_prim.GetTypeName() == expected_types[i % len(expected_types)]

    sim.reset()
    physics_sim_view = sim.physics_manager.get_physics_sim_view()
    physx_view = physics_sim_view.create_rigid_body_view("/World/envs/env_*/Object")
    for _ in range(100):
        sim.step()
    transforms = wp.to_torch(physx_view.get_transforms())
    distance_from_origin = torch.linalg.norm(transforms[:, :2], dim=-1)
    assert torch.all(distance_from_origin < 0.1)


def test_colocation_collision_filter_homogeneous(sim):
    """Verify colocated clones of a single prototype stay stable after PhysX cloning.

    All clones are spawned at exactly the same pose; if the collision filter is wrong the pile
    explodes on reset. This asserts the filter keeps the colocated objects stable while stepping
    across CPU and CUDA backends.
    """
    _run_colocation_collision_filter(
        sim,
        sim_utils.ConeCfg(
            radius=0.3,
            height=0.6,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
            mass_props=sim_utils.MassCfg(mass=100.0),
            rigid_props=PhysxRigidBodyCfg(solver_position_iteration_count=4, solver_velocity_iteration_count=0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        expected_types=["Cone"],
        assert_count=True,
    )


def _run_sphere_velocity_sim(sim, use_physx_replicate: bool, num_steps: int = 10) -> torch.Tensor:
    """Run a 2-env sphere simulation and return the full velocity trajectory.

    Returns a (num_steps, num_envs, 6) tensor of velocities at each step.
    """
    num_envs = 2
    spacing = 5.0
    stage = sim_utils.get_current_stage()

    sim_utils.create_prim("/World/envs", "Xform")
    sim_utils.create_prim("/World/envs/env_0", "Xform")

    sphere_cfg = sim_utils.SphereCfg(
        radius=0.25,
        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
        mass_props=sim_utils.MassCfg(mass=0.5),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
    )
    sphere_cfg.func("/World/envs/env_0/ball", sphere_cfg, translation=(0.0, 0.0, 0.5))

    env_ids = np.arange(num_envs, dtype=np.int64)
    positions = np.array([[0.0, 0.0, 0.0], [spacing, 0.0, 0.0]], dtype=np.float32)
    mapping = np.ones((1, num_envs), dtype=np.bool_)

    if use_physx_replicate:
        physx_replicate(
            stage,
            sources=["/World/envs/env_0/ball"],
            destinations=["/World/envs/env_{}/ball"],
            env_ids=env_ids,
            mapping=mapping,
        )

    usd_replicate(
        stage,
        sources=["/World/envs/env_0"],
        destinations=["/World/envs/env_{}"],
        env_ids=env_ids,
        mask=mapping,
        positions=positions,
    )

    sim.reset()

    physics_sim_view = sim.physics_manager.get_physics_sim_view()
    ball_view = physics_sim_view.create_rigid_body_view("/World/envs/env_*/ball")
    assert ball_view.count == num_envs, f"Expected {num_envs} balls, got {ball_view.count}"

    device = sim.cfg.device
    vel = wp.from_torch(torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * num_envs, dtype=torch.float32, device=device))
    indices = wp.from_torch(torch.arange(num_envs, dtype=torch.int32, device=device))

    velocities = []
    for _ in range(num_steps):
        ball_view.set_velocities(vel, indices)
        sim.step()
        v = wp.to_torch(ball_view.get_velocities())
        velocities.append(v.cpu().clone())

    return torch.stack(velocities)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU_AND_DEFAULT_CUDA))
def test_physx_replicate_vs_no_replicate(device):
    """Test that physx_replicate does not change the physics behavior of env_0.

    With ``attach_fn`` excluding ``/World/envs``, env_0 receives its physics body
    from the replicator (as the source of ``rep.replicate()``) rather than from
    normal USD parsing; the resulting trajectory must match the USD-parsed baseline,
    and the replicated env_1 must match env_0.
    """
    with build_simulation_context(device=device, dt=0.01, add_lighting=False) as sim_no_rep:
        baseline = _run_sphere_velocity_sim(sim_no_rep, use_physx_replicate=False)

    with build_simulation_context(device=device, dt=0.01, add_lighting=False) as sim_rep:
        with_rep = _run_sphere_velocity_sim(sim_rep, use_physx_replicate=True)

    for idx in range(baseline.shape[0]):
        diff = (with_rep[idx, 0] - baseline[idx, 0]).abs().max().item()
        assert diff < 1e-3, f"step {idx}: replicate vs no-replicate diverge, max_diff={diff}"
        diff = (with_rep[idx, 0] - with_rep[idx, 1]).abs().max().item()
        assert diff < 1e-3, f"step {idx}: env_0 and env_1 diverge, max_diff={diff}"


@cuda_only
def test_disabled_fabric_change_notifies_toggles_ifabricusd_flag(sim):
    """Regression: ``disabled_fabric_change_notifies`` actually toggles the IFabricUsd flag.

    The PR's perf win depends on ``setEnableChangeNotifies`` being driven correctly by the
    ctypes binding in ``_fabric_notices.py``. That binding reads hardcoded vtable offsets
    and could silently no-op if Kit's ABI shifts (offsets drift) or libcarb fails to load.

    A perf-delta assertion can't be done reliably in synthetic isolation — the listener's
    cost only shows up under full Kit+PhysX integration paths that this test environment
    doesn't reproduce; production-scene benchmarks are the PR's load-bearing perf evidence.
    What this test guards is the mechanic itself: ``is_enabled`` flips on entry, restores
    on exit when ``restore=True``, stays off when ``restore=False``, and re-entrant nested
    blocks behave correctly.
    """
    import usdrt
    from pxr import UsdUtils

    bindings = _fabric_notices.get_bindings()
    if bindings is None:
        pytest.skip("omni::fabric::IFabricUsd unavailable — Fabric notice path inert here")

    stage = sim_utils.get_current_stage()
    cache = UsdUtils.StageCache.Get()
    cached_id = cache.GetId(stage)
    stage_id = cached_id.ToLongInt() if cached_id.IsValid() else cache.Insert(stage).ToLongInt()
    fabric_id = usdrt.Usd.Stage.Attach(stage_id).GetFabricId().id

    # 1. Listener starts enabled.
    assert bindings.is_enabled(fabric_id), "Fabric notice listener should be enabled at test start"

    # 2. Default ``restore=True`` round-trips the flag.
    with disabled_fabric_change_notifies(stage):
        assert not bindings.is_enabled(fabric_id), "listener should be suspended inside the with block"
    assert bindings.is_enabled(fabric_id), "listener should be restored on exit when restore=True"

    # 3. ``restore=False`` leaves the flag off. Manually re-enable to get back to a
    #    known state for subsequent assertions.
    with disabled_fabric_change_notifies(stage, restore=False):
        assert not bindings.is_enabled(fabric_id), "listener should be suspended inside the with block"
    assert not bindings.is_enabled(fabric_id), "listener should remain suspended on exit when restore=False"
    bindings.set_enable(fabric_id, True)
    assert bindings.is_enabled(fabric_id)

    # 4. Re-entrant nesting: inner exits don't re-enable while outer still wants it suspended.
    with disabled_fabric_change_notifies(stage):
        assert not bindings.is_enabled(fabric_id)
        with disabled_fabric_change_notifies(stage):
            assert not bindings.is_enabled(fabric_id)
        assert not bindings.is_enabled(fabric_id), "inner exit must not re-enable while outer is active"
    assert bindings.is_enabled(fabric_id), "outer exit should restore the flag"
