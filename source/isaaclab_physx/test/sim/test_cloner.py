# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for PhysX-dependent cloner utilities."""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest
import torch
import warp as wp
from isaaclab_physx.cloner import physx_replicate

import isaaclab.sim as sim_utils
from isaaclab.cloner import TemplateCloneCfg, clone_from_template, sequential, usd_replicate
from isaaclab.sim import build_simulation_context


@pytest.fixture(params=["cpu", "cuda"])
def sim(request):
    """Provide a fresh simulation context for each test on CPU and CUDA."""
    with build_simulation_context(device=request.param, dt=0.01, add_lighting=False) as sim:
        yield sim


def test_physx_replicate_no_error(sim):
    """PhysX replicator call runs without raising exceptions for simple mapping."""
    sim_utils.create_prim("/World/envs", "Xform")
    sim_utils.create_prim("/World/template", "Xform")
    sim_utils.create_prim("/World/template/A", "Xform")

    num_envs = 2
    env_ids = torch.arange(num_envs, dtype=torch.long)
    for i in range(num_envs):
        sim_utils.create_prim(f"/World/envs/env_{i}", "Xform")

    mapping = torch.ones((1, num_envs), dtype=torch.bool)

    physx_replicate(
        sim_utils.get_current_stage(),
        sources=["/World/template/A"],
        destinations=["/World/envs/env_{}/A"],
        env_ids=env_ids,
        mapping=mapping,
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
    with patch("isaaclab_physx.cloner.physx_replicate.get_physx_replicator_interface", return_value=mock_rep):
        physx_replicate(
            stage,
            sources=[src],
            destinations=["/World/envs/env_{}"],
            env_ids=torch.arange(num_envs, dtype=torch.long),
            mapping=torch.ones((1, num_envs), dtype=torch.bool),
        )

    assert replicate_calls == expected_worlds, (
        f"Expected replicate world counts {expected_worlds}, got {replicate_calls}"
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_physx_replicate_isolated_source_loaded_without_replication(sim, device):
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
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    sphere_cfg.func("/World/envs/env_0/Sphere", sphere_cfg)

    physx_replicate(
        stage,
        sources=["/World/envs/env_0/Sphere"],
        destinations=["/World/envs/env_{}/Sphere"],
        env_ids=torch.tensor([0], dtype=torch.long),
        mapping=torch.ones((1, 1), dtype=torch.bool),
        device=device,
    )

    sim.reset()

    physics_sim_view = sim.physics_manager.get_physics_sim_view()
    physx_view = physics_sim_view.create_rigid_body_view("/World/envs/env_*/Sphere")
    assert physx_view is not None and physx_view.count == 1, (
        f"Expected 1 rigid body at /World/envs/env_0/Sphere, got {'None' if physx_view is None else physx_view.count}."
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_physx_replicate_heterogeneous_isolated_sources(sim, device):
    """physx_replicate handles heterogeneous sources excluding self from world lists.

    This is the Dexsuite scenario: multiple object types, each with a designated proto-env.
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

    mapping = torch.zeros((3, num_envs), dtype=torch.bool)
    mapping[0, [0, 2, 4]] = True
    mapping[1, [5]] = True
    mapping[2, [7, 11]] = True

    mock_rep, replicate_calls, attach_excluded = _make_mock_physx_rep_detailed()
    with patch("isaaclab_physx.cloner.physx_replicate.get_physx_replicator_interface", return_value=mock_rep):
        physx_replicate(
            stage,
            sources=["/World/envs/env_0/Object", "/World/envs/env_5/Object", "/World/envs/env_7/Object"],
            destinations=["/World/envs/env_{}/Object"] * 3,
            env_ids=torch.arange(num_envs, dtype=torch.long),
            mapping=mapping,
            device=device,
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


def test_clone_from_template(sim):
    """Clone prototypes via TemplateCloneCfg and clone_from_template and exercise both USD and PhysX.

    Steps:
    - Create /World/template and /World/envs/env_0..env_31
    - Spawn three prototypes under /World/template/Object/proto_asset_.*
    - Clone using TemplateCloneCfg with random_heterogeneous_cloning=False (modulo mapping)
    - Verify modulo placement exists; then call sim.reset(), and create PhysX view
    """
    num_clones = 32
    clone_cfg = TemplateCloneCfg(device=sim.cfg.device, clone_strategy=sequential)
    sim_utils.create_prim(clone_cfg.template_root, "Xform")
    sim_utils.create_prim(f"{clone_cfg.template_root}/Object", "Xform")
    sim_utils.create_prim("/World/envs", "Xform")
    for i in range(num_clones):
        sim_utils.create_prim(f"/World/envs/env_{i}", "Xform", translation=(0, 0, 0))

    cfg = sim_utils.MultiAssetSpawnerCfg(
        assets_cfg=[
            sim_utils.ConeCfg(
                radius=0.3,
                height=0.6,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
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
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    prim = cfg.func(f"{clone_cfg.template_root}/Object/{clone_cfg.template_prototype_identifier}_.*", cfg)
    assert prim.IsValid()

    stage = sim_utils.get_current_stage()
    clone_from_template(stage, num_clones=num_clones, template_clone_cfg=clone_cfg)

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
    object_view_regex = f"{clone_cfg.clone_regex}/Object".replace(".*", "*")
    physics_sim_view = sim.physics_manager.get_physics_sim_view()
    physx_view = physics_sim_view.create_rigid_body_view(object_view_regex)
    assert physx_view is not None


def _run_colocation_collision_filter(sim, asset_cfg, expected_types, assert_count=False):
    """Shared harness for colocated collision filter checks across devices."""
    num_clones = 32
    clone_cfg = TemplateCloneCfg(device=sim.cfg.device, clone_strategy=sequential)
    sim_utils.create_prim(clone_cfg.template_root, "Xform")
    sim_utils.create_prim(f"{clone_cfg.template_root}/Object", "Xform")
    sim_utils.create_prim("/World/envs", "Xform")
    for i in range(num_clones):
        sim_utils.create_prim(f"/World/envs/env_{i}", "Xform", translation=(0, 0, 0))

    prim = asset_cfg.func(f"{clone_cfg.template_root}/Object/{clone_cfg.template_prototype_identifier}_.*", asset_cfg)
    assert prim.IsValid()

    stage = sim_utils.get_current_stage()
    clone_from_template(stage, num_clones=num_clones, template_clone_cfg=clone_cfg)

    primitive_prims = sim_utils.get_all_matching_child_prims(
        "/World/envs", predicate=lambda prim: prim.GetTypeName() in expected_types
    )

    if assert_count:
        assert len(primitive_prims) == num_clones

    for i, primitive_prim in enumerate(primitive_prims):
        assert primitive_prim.GetTypeName() == expected_types[i % len(expected_types)]

    sim.reset()
    object_view_regex = f"{clone_cfg.clone_regex}/Object".replace(".*", "*")
    physics_sim_view = sim.physics_manager.get_physics_sim_view()
    physx_view = physics_sim_view.create_rigid_body_view(object_view_regex)
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
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        expected_types=["Cone"],
        assert_count=True,
    )


@pytest.mark.xfail(reason="Heterogeneous cloning with collision filtering not yet fully supported")
def test_colocation_collision_filter_heterogeneous(sim):
    """Verify colocated clones of multiple prototypes retain modulo ordering and remain stable.

    The cone, cube, and sphere are all spawned in the identical pose for every clone; an incorrect
    collision filter would blow up the simulation on reset. This guards both modulo ordering and
    that the colocated set stays stable through PhysX steps across CPU and CUDA.
    """
    _run_colocation_collision_filter(
        sim,
        sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.ConeCfg(
                    radius=0.3,
                    height=0.6,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
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
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        expected_types=["Cone", "Cube", "Sphere"],
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
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    sphere_cfg.func("/World/envs/env_0/ball", sphere_cfg, translation=(0.0, 0.0, 0.5))

    env_ids = torch.arange(num_envs, dtype=torch.long)
    positions = torch.tensor([[0.0, 0.0, 0.0], [spacing, 0.0, 0.0]])
    mapping = torch.ones((1, num_envs), dtype=torch.bool)

    if use_physx_replicate:
        physx_replicate(
            stage,
            sources=["/World/envs/env_0/ball"],
            destinations=["/World/envs/env_{}/ball"],
            env_ids=env_ids,
            mapping=mapping,
            device=sim.cfg.device,
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


@pytest.mark.isaacsim_ci
def test_physx_replicate_env_consistency(sim):
    """Test that env_0 and env_1 produce matching velocities when using physx_replicate."""
    trajectory = _run_sphere_velocity_sim(sim, use_physx_replicate=True)

    for idx in range(trajectory.shape[0]):
        v0 = trajectory[idx, 0]
        v1 = trajectory[idx, 1]
        diff = (v0 - v1).abs().max().item()
        assert diff < 1e-3, f"step {idx}: env_0 and env_1 diverge, max_diff={diff}"


@pytest.mark.xfail(reason="Source env gets physics from replicator, not USD parsing; may diverge from baseline.")
@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_physx_replicate_vs_no_replicate(device):
    """Test that physx_replicate does not change the physics behavior of env_0.

    With ``attach_fn`` excluding ``/World/envs``, env_0 receives its physics body
    from the replicator (as the source of ``rep.replicate()``) rather than from
    normal USD parsing, which may produce subtly different behaviour.
    """
    with build_simulation_context(device=device, dt=0.01, add_lighting=False) as sim_no_rep:
        baseline = _run_sphere_velocity_sim(sim_no_rep, use_physx_replicate=False)

    with build_simulation_context(device=device, dt=0.01, add_lighting=False) as sim_rep:
        with_rep = _run_sphere_velocity_sim(sim_rep, use_physx_replicate=True)

    for idx in range(baseline.shape[0]):
        diff = (with_rep[idx, 0] - baseline[idx, 0]).abs().max().item()
        assert diff < 1e-3, f"step {idx}: replicate vs no-replicate diverge, max_diff={diff}"
