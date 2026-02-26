# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for cloner utilities and InteractiveScene cloning behavior."""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest
import torch
import warp as wp
from isaaclab_physx.cloner import physx_replicate

from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.cloner import TemplateCloneCfg, clone_from_template, sequential, usd_replicate
from isaaclab.sim import build_simulation_context


@pytest.fixture(params=["cpu", "cuda"])
def sim(request):
    """Provide a fresh simulation context for each test on CPU and CUDA."""
    with build_simulation_context(device=request.param, dt=0.01, add_lighting=False) as sim:
        yield sim


def test_usd_replicate_with_positions_and_mask(sim):
    """Replicate sources to selected envs and author translate ops from positions."""
    # Prepare sources under /World/template
    sim_utils.create_prim("/World/template", "Xform")
    sim_utils.create_prim("/World/template/A", "Xform")
    sim_utils.create_prim("/World/template/B", "Xform")

    # Prepare destination env namespaces
    num_envs = 3
    env_ids = torch.arange(num_envs, dtype=torch.long)
    sim_utils.create_prim("/World/envs", "Xform")
    for i in range(num_envs):
        sim_utils.create_prim(f"/World/envs/env_{i}", "Xform")

    # Map A -> env 0 and 2; B -> env 1 only
    mask = torch.zeros((2, num_envs), dtype=torch.bool)
    mask[0, [0, 2]] = True
    mask[1, [1]] = True

    usd_replicate(
        sim_utils.get_current_stage(),
        sources=["/World/template/A", "/World/template/B"],
        destinations=["/World/envs/env_{}/Object/A", "/World/envs/env_{}/Object/B"],
        env_ids=env_ids,
        mask=mask,
    )

    # Validate replication and translate op
    stage = sim_utils.get_current_stage()
    assert stage.GetPrimAtPath("/World/envs/env_0/Object/A").IsValid()
    assert not stage.GetPrimAtPath("/World/envs/env_0/Object/B").IsValid()
    assert stage.GetPrimAtPath("/World/envs/env_1/Object/B").IsValid()
    assert not stage.GetPrimAtPath("/World/envs/env_1/Object/A").IsValid()
    assert stage.GetPrimAtPath("/World/envs/env_2/Object/A").IsValid()

    # Check xformOp:translate authored for env_2/A
    prim = stage.GetPrimAtPath("/World/envs/env_2/Object/A")
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    assert any(op.GetOpType() == UsdGeom.XformOp.TypeTranslate for op in ops)


def test_usd_replicate_depth_order_parent_child(sim):
    """Replicate parent and child when provided out of order; parent should exist before child."""
    # Prepare sources
    sim_utils.create_prim("/World/template", "Xform")
    sim_utils.create_prim("/World/template/Parent", "Xform")
    sim_utils.create_prim("/World/template/Parent/Child", "Xform")

    # Destinations (single env)
    env_ids = torch.tensor([0, 1], dtype=torch.long)
    sim_utils.create_prim("/World/envs", "Xform")
    sim_utils.create_prim("/World/envs/env_0", "Xform")
    sim_utils.create_prim("/World/envs/env_1", "Xform")

    # Provide child first, then parent; depth sort should handle this
    usd_replicate(
        sim_utils.get_current_stage(),
        sources=["/World/template/Parent/Child", "/World/template/Parent"],
        destinations=["/World/envs/env_{}/Parent/Child", "/World/envs/env_{}/Parent"],
        env_ids=env_ids,
    )

    stage = sim_utils.get_current_stage()
    for i in range(2):
        assert stage.GetPrimAtPath(f"/World/envs/env_{i}/Parent").IsValid()
        assert stage.GetPrimAtPath(f"/World/envs/env_{i}/Parent/Child").IsValid()


def test_physx_replicate_no_error(sim):
    """PhysX replicator call runs without raising exceptions for simple mapping."""
    # Prepare sources and envs
    sim_utils.create_prim("/World/envs", "Xform")
    sim_utils.create_prim("/World/template", "Xform")
    sim_utils.create_prim("/World/template/A", "Xform")

    num_envs = 2
    env_ids = torch.arange(num_envs, dtype=torch.long)
    for i in range(num_envs):
        sim_utils.create_prim(f"/World/envs/env_{i}", "Xform")

    mapping = torch.ones((1, num_envs), dtype=torch.bool)

    # Should not raise
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
        # source == env_0; 3 envs → self-world excluded, replicate to env_1 and env_2 only (2 worlds)
        (3, "/World/envs/env_0", [2]),
        # source == env_0; 1 env → self-excluded → worlds=[] → rep.replicate skipped.
        # env_0 was already registered by the replicator's stage parse (attach_fn returns only
        # the template path, so env_0 is not excluded and is parsed as a simulation body).
        (1, "/World/envs/env_0", []),
        # source path does not match template (external template) → all envs included
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

    # Spawn an isolated rigid body at env_0 (the only env — self is the sole destination).
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

    # Start simulation — PhysX builds its internal scene from the USD stage.
    sim.reset()

    # The rigid body at env_0/Sphere must be findable (first instance was loaded by stage parse).
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
    mapping[0, [0, 2, 4]] = True  # env_0/Object: multi-world (includes self)
    mapping[1, [5]] = True  # env_5/Object: isolated (only self)
    mapping[2, [7, 11]] = True  # env_7/Object: multi-world (includes self)

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

    # env_5 is isolated → skipped; env_0 → 2 worlds; env_7 → 1 world
    expected = [
        ("/World/envs/env_0/Object", 2),
        ("/World/envs/env_7/Object", 1),
    ]
    assert replicate_calls == expected, (
        f"Expected {expected}, got {replicate_calls}. "
        "env_5/Object (isolated, worlds=[5]) must be skipped — it is registered by the "
        "replicator's stage parse, not by rep.replicate."
    )


def test_usd_replicate_self_copy_skips_copy_spec(sim):
    """usd_replicate must not call Sdf.CopySpec when source and destination paths are identical.

    Sdf.CopySpec(src, src) is a no-op in the current USD version so it does not corrupt children,
    but the call is still wasteful. The guard ensures it is skipped entirely. This test mocks
    Sdf.CopySpec to verify it is called exactly once (for env_1) and never for the self case (env_0).
    """
    from unittest.mock import patch

    import isaaclab.cloner.cloner_utils as _cloner_mod

    stage = sim_utils.get_current_stage()
    sim_utils.create_prim("/World/envs", "Xform")
    sim_utils.create_prim("/World/envs/env_0", "Xform")
    sim_utils.create_prim("/World/envs/env_0/Robot", "Xform")
    sim_utils.create_prim("/World/envs/env_0/Robot/base_link", "Xform")
    sim_utils.create_prim("/World/envs/env_1", "Xform")

    copy_calls: list[tuple[str, str]] = []
    real_copy_spec = _cloner_mod.Sdf.CopySpec

    def capturing_copy_spec(src_layer, src_path, dst_layer, dst_path):
        copy_calls.append((str(src_path), str(dst_path)))
        return real_copy_spec(src_layer, src_path, dst_layer, dst_path)

    with patch.object(_cloner_mod.Sdf, "CopySpec", capturing_copy_spec):
        usd_replicate(
            stage,
            sources=["/World/envs/env_0"],
            destinations=["/World/envs/env_{}"],
            env_ids=torch.tensor([0, 1], dtype=torch.long),
            mask=torch.ones((1, 2), dtype=torch.bool),
        )

    # CopySpec must be called for env_1 but never for env_0 (self-copy)
    assert all(src != dst for src, dst in copy_calls), f"Self-copy detected in CopySpec calls: {copy_calls}"
    assert any(dst == "/World/envs/env_1" for _, dst in copy_calls), "CopySpec was not called for env_1"


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
    sim_utils.create_prim(f"{clone_cfg.template_root}/Object", "Xform")  # Parent for prototypes
    sim_utils.create_prim("/World/envs", "Xform")
    for i in range(num_clones):
        sim_utils.create_prim(f"/World/envs/env_{i}", "Xform", translation=(0, 0, 0))

    # Spawn prototypes under template
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

    # Exercise PhysX initialization; should not raise error
    sim.reset()
    object_view_regex = f"{clone_cfg.clone_regex}/Object".replace(".*", "*")
    physics_sim_view = sim.physics_manager.get_physics_sim_view()
    physx_view = physics_sim_view.create_rigid_body_view(object_view_regex)
    assert physx_view is not None


@pytest.mark.parametrize(
    "parent_paths, spawn_pattern, expected_child_paths, bad_path, match_expr",
    [
        # Case A – single wildcard in root_path.
        # The child must appear under every matching parent (not just the first).
        # Old code spawned at the wrong synthetic path /World/rig_00/Sensor.
        (
            ["/World/rig_0_alpha", "/World/rig_0_beta", "/World/rig_0_gamma"],
            "/World/rig_0_.*/Sensor",
            ["/World/rig_0_alpha/Sensor", "/World/rig_0_beta/Sensor", "/World/rig_0_gamma/Sensor"],
            "/World/rig_00/Sensor",
            "/World/rig_0_.*",
        ),
        # Case A – double wildcard in root_path (multiple ".*" levels).
        # Old code replaced BOTH ".*" with 0, giving the wrong path /World/group_0/slot_0/Sensor.
        (
            [
                "/World/group_a/slot_0",
                "/World/group_a/slot_1",
                "/World/group_b/slot_0",
                "/World/group_b/slot_1",
            ],
            "/World/group_.*/slot_.*/Sensor",
            [
                "/World/group_a/slot_0/Sensor",
                "/World/group_a/slot_1/Sensor",
                "/World/group_b/slot_0/Sensor",
                "/World/group_b/slot_1/Sensor",
            ],
            "/World/group_0/slot_0/Sensor",
            "/World/group_.*/slot_.*",
        ),
        # Case B – wildcard only in asset_path (leaf), no wildcard in root_path.
        # This is the proto_asset_* pattern: no matching prims exist yet; the decorator
        # must create a single prototype named proto_0 under the one real parent.
        # root_path has no regex so source_prim_paths == [root_path] and no copy step runs.
        (
            ["/World/template/Object"],
            "/World/template/Object/proto_.*",
            ["/World/template/Object/proto_0"],
            "/World/template/Object0/proto_0",  # spurious parent that must NOT be created
            "/World/template/Object",
        ),
    ],
)
def test_clone_decorator_wildcard_patterns(sim, parent_paths, spawn_pattern, expected_child_paths, bad_path, match_expr):
    """The @clone decorator handles two distinct wildcard patterns correctly.

    Case A – ``.*`` in root_path (parent is a regex): the child prim is spawned at
    ``source_prim_paths[0]`` as a prototype and then copied to every other matching
    parent via ``Sdf.CopySpec``, so **all** parents end up with the child.  The old
    ``prim_path.replace(".*", "0")`` approach created spurious intermediate prims
    that inflated ``find_matching_prims`` counts and broke tiled-camera initialization.

    Case B – ``.*`` only in asset_path (leaf): no parent regex, so
    ``source_prim_paths == [root_path]`` (one entry, no copy step).  Replacing
    ``".*"`` → ``"0"`` in the asset name gives the intended prototype name
    (e.g. ``proto_asset_0``) under the single real parent.
    """
    for path in parent_paths:
        sim_utils.create_prim(path, "Xform")

    cfg = sim_utils.ConeCfg(radius=0.1, height=0.2)
    cfg.func(spawn_pattern, cfg)

    stage = sim_utils.get_current_stage()

    # Every expected child path must exist
    for child_path in expected_child_paths:
        assert stage.GetPrimAtPath(child_path).IsValid(), (
            f"Prim was not spawned at '{child_path}'. "
            "The @clone decorator may have used the wrong spawn path."
        )

    # The spurious path from the old replace(".*", "0") must NOT exist
    assert not stage.GetPrimAtPath(bad_path).IsValid(), (
        f"Spurious prim found at '{bad_path}'. "
        "The @clone decorator incorrectly derived the spawn path by replacing '.*' with '0'."
    )

    # find_matching_prims must see exactly the original parents — no spurious extras
    all_matching = sim_utils.find_matching_prims(match_expr)
    assert len(all_matching) == len(parent_paths), (
        f"Expected {len(parent_paths)} matching prims, got {len(all_matching)}. "
        "Spurious parent prims were likely created by the @clone decorator."
    )


def _run_colocation_collision_filter(sim, asset_cfg, expected_types, assert_count=False):
    """Shared harness for colocated collision filter checks across devices."""
    num_clones = 32
    clone_cfg = TemplateCloneCfg(device=sim.cfg.device, clone_strategy=sequential)
    sim_utils.create_prim(clone_cfg.template_root, "Xform")
    sim_utils.create_prim(f"{clone_cfg.template_root}/Object", "Xform")  # Parent for prototypes
    sim_utils.create_prim("/World/envs", "Xform")
    for i in range(num_clones):
        sim_utils.create_prim(f"/World/envs/env_{i}", "Xform", translation=(0, 0, 0))

    # ".*" is in the asset leaf name only (Case B): no parent regex, so the decorator
    # creates a single prototype named proto_asset_0 under the one real parent.
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
