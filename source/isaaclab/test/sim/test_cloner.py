# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for USD cloner utilities (no PhysX dependency)."""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest
import torch

from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.cloner import ClonePlan, TemplateCloneCfg, clone_from_template, sequential, usd_replicate
from isaaclab.sim import build_simulation_context

pytestmark = pytest.mark.isaacsim_ci


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


@pytest.mark.parametrize(
    "parent_paths, spawn_pattern, expected_child_paths, bad_path, match_expr",
    [
        (
            ["/World/rig_0_alpha", "/World/rig_0_beta", "/World/rig_0_gamma"],
            "/World/rig_0_.*/Sensor",
            ["/World/rig_0_alpha/Sensor", "/World/rig_0_beta/Sensor", "/World/rig_0_gamma/Sensor"],
            "/World/rig_00/Sensor",
            "/World/rig_0_.*",
        ),
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
        (
            ["/World/template/Object"],
            "/World/template/Object/proto_.*",
            ["/World/template/Object/proto_0"],
            "/World/template/Object0/proto_0",
            "/World/template/Object",
        ),
    ],
)
def test_clone_decorator_wildcard_patterns(
    sim, parent_paths, spawn_pattern, expected_child_paths, bad_path, match_expr
):
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
            f"Prim was not spawned at '{child_path}'. The @clone decorator may have used the wrong spawn path."
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


def test_clone_from_template_returns_clone_plan(sim):
    """clone_from_template exposes per-group ClonePlan dicts with prototype-to-env masks.

    Builds two USD prototypes under one group, clones across four envs with the deterministic
    sequential strategy, and asserts the returned dict has one entry keyed by the group's
    destination template, with a ``[2, 4]`` boolean mask whose columns sum to one.
    """
    num_clones = 4
    cfg = TemplateCloneCfg(device=sim.cfg.device, clone_strategy=sequential, clone_physics=False)

    sim_utils.create_prim(cfg.template_root, "Xform")
    sim_utils.create_prim(f"{cfg.template_root}/Object", "Xform")
    sim_utils.create_prim(f"{cfg.template_root}/Object/proto_asset_0", "Xform")
    sim_utils.create_prim(f"{cfg.template_root}/Object/proto_asset_1", "Xform")
    sim_utils.create_prim("/World/envs", "Xform")
    for i in range(num_clones):
        sim_utils.create_prim(f"/World/envs/env_{i}", "Xform", translation=(0, 0, 0))

    stage = sim_utils.get_current_stage()
    plans = clone_from_template(stage, num_clones=num_clones, template_clone_cfg=cfg)

    assert isinstance(plans, dict)
    assert list(plans.keys()) == ["/World/envs/env_{}/Object"]
    plan = plans["/World/envs/env_{}/Object"]
    assert isinstance(plan, ClonePlan)
    assert plan.dest_template == "/World/envs/env_{}/Object"
    assert sorted(plan.prototype_paths) == [
        "/World/template/Object/proto_asset_0",
        "/World/template/Object/proto_asset_1",
    ]
    assert plan.clone_mask.shape == (2, num_clones)
    assert plan.clone_mask.dtype == torch.bool
    # Each env gets exactly one prototype (column-sum invariant)
    assert torch.all(plan.clone_mask.sum(dim=0) == 1)
    # Sequential strategy assigns env i → prototype (i % num_protos)
    actual_proto_idx = plan.clone_mask.to(torch.int).argmax(dim=0).cpu()
    assert torch.equal(actual_proto_idx, torch.tensor([0, 1, 0, 1]))


def test_clone_from_template_returns_empty_dict_when_no_prototypes(sim):
    """clone_from_template returns an empty dict when no prototypes match the identifier."""
    num_clones = 2
    cfg = TemplateCloneCfg(device=sim.cfg.device, clone_strategy=sequential, clone_physics=False)

    sim_utils.create_prim(cfg.template_root, "Xform")
    sim_utils.create_prim("/World/envs", "Xform")
    for i in range(num_clones):
        sim_utils.create_prim(f"/World/envs/env_{i}", "Xform", translation=(0, 0, 0))

    stage = sim_utils.get_current_stage()
    plans = clone_from_template(stage, num_clones=num_clones, template_clone_cfg=cfg)

    assert plans == {}
