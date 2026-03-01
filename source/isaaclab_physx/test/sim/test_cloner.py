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
from isaaclab.cloner import TemplateCloneCfg, clone_from_template, sequential
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
def test_physx_replicate_excludes_self_world(sim, num_envs, src, expected_worlds):
    """physx_replicate skips rep.replicate when all worlds are self-copies.

    attach_fn returns only the template path so the replicator's stage parse registers all
    existing env prims as simulation bodies. rep.replicate is only called for the non-self
    replica worlds. When filtering removes all worlds (isolated single-env case, or global
    num_envs=1), rep.replicate is skipped entirely — the source prim is already registered
    by the stage parse and no replication is needed.
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
    """An isolated source (worlds=[self]) is registered by the stage parse, not rep.replicate.

    When physx_replicate encounters a source whose only world is itself, it skips rep.replicate
    (self-exclusion → empty worlds list → continue). The prim must already be in the stage
    (put there by proto_mask usd_replicate) so the replicator's stage parse picks it up as a
    simulation body. This test verifies:
      1. rep.replicate is NOT called for the isolated source (no self-copy).
      2. After sim.reset(), PhysX can find the rigid body at the isolated env path — i.e. the
         first instance IS loaded even though rep.replicate was never invoked for it.
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
        f"Expected 1 rigid body at /World/envs/env_0/Sphere, got "
        f"{'None (prim not found by PhysX)' if physx_view is None else physx_view.count}. "
        "Isolated source (worlds=[self]) must be registered by the stage parse when "
        "rep.replicate is skipped — verify attach_fn does not exclude env prim paths."
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_physx_replicate_heterogeneous_isolated_sources(sim, device):
    """physx_replicate handles heterogeneous sources where some map only to themselves.

    This is the Dexsuite scenario: multiple object types, each with a designated proto-env.
    Some types are assigned to only one environment (themselves), making them 'isolated'.
    Isolated sources must be skipped by rep.replicate — they are already registered by the
    replicator's stage parse (attach_fn does not exclude env paths). Multi-world sources
    must replicate only to their non-self worlds.

    Sources and expected behaviour:
      env_0/Object → worlds [0, 2, 4]   → self-excluded → replicate to [2, 4]  (2 worlds)
      env_5/Object → worlds [5]          → self-excluded → skip (isolated)       (0 worlds)
      env_7/Object → worlds [7, 11]      → self-excluded → replicate to [11]    (1 world)
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

    mock_rep, replicate_calls, _ = _make_mock_physx_rep_detailed()
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
        ("/World/envs/env_7/Object", 1),
    ]
    assert replicate_calls == expected, (
        f"Expected {expected}, got {replicate_calls}. "
        "env_5/Object (isolated, worlds=[5]) must be skipped — it is registered by the "
        "replicator's stage parse, not by rep.replicate."
    )


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
