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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.cloner import (
    REPLICATION_QUEUE,
    ClonePlan,
    UsdReplicateContext,
    grid_transforms,
    iter_clone_plan_matches,
    make_clone_plan,
    queue_usd_replication,
    replicate,
    resolve_clone_plan_source,
    sequential,
    usd_replicate,
)
from isaaclab.sim import build_simulation_context

pytestmark = pytest.mark.isaacsim_ci


@pytest.fixture(params=["cpu", "cuda"])
def sim(request):
    """Provide a fresh simulation context for each test on CPU and CUDA."""
    with build_simulation_context(device=request.param, dt=0.01, add_lighting=False) as sim:
        yield sim


@pytest.fixture(autouse=True)
def _drain_replication_queue():
    """Ensure REPLICATION_QUEUE starts empty for every test and is cleared after."""
    REPLICATION_QUEUE.clear()
    yield
    REPLICATION_QUEUE.clear()


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


def test_usd_replicate_context_queue_and_replicate(sim):
    """UsdReplicateContext queues copy specs and applies them on replicate."""
    sim_utils.create_prim("/World/template", "Xform")
    sim_utils.create_prim("/World/template/A", "Xform")
    sim_utils.create_prim("/World/envs", "Xform")
    sim_utils.create_prim("/World/envs/env_0", "Xform")
    sim_utils.create_prim("/World/envs/env_1", "Xform")

    stage = sim_utils.get_current_stage()
    ctx = UsdReplicateContext(stage)
    ctx.queue_mapping(
        sources=["/World/template/A"],
        destinations=["/World/envs/env_{}/A"],
        env_ids=torch.tensor([0, 1], dtype=torch.long),
    )
    assert not stage.GetPrimAtPath("/World/envs/env_1/A").IsValid()
    ctx.replicate()

    assert stage.GetPrimAtPath("/World/envs/env_0/A").IsValid()
    assert stage.GetPrimAtPath("/World/envs/env_1/A").IsValid()


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
    """usd_replicate must not call Sdf.CopySpec when source and destination paths are identical."""
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
    """The @clone decorator handles two distinct wildcard patterns correctly."""
    for path in parent_paths:
        sim_utils.create_prim(path, "Xform")

    cfg = sim_utils.ConeCfg(radius=0.1, height=0.2)
    cfg.func(spawn_pattern, cfg)

    stage = sim_utils.get_current_stage()

    for child_path in expected_child_paths:
        assert stage.GetPrimAtPath(child_path).IsValid(), (
            f"Prim was not spawned at '{child_path}'. The @clone decorator may have used the wrong spawn path."
        )

    assert not stage.GetPrimAtPath(bad_path).IsValid(), (
        f"Spurious prim found at '{bad_path}'. "
        "The @clone decorator incorrectly derived the spawn path by replacing '.*' with '0'."
    )

    all_matching = sim_utils.find_matching_prims(match_expr)
    assert len(all_matching) == len(parent_paths), (
        f"Expected {len(parent_paths)} matching prims, got {len(all_matching)}. "
        "Spurious parent prims were likely created by the @clone decorator."
    )


def test_queue_usd_replication_only_appends(sim):
    """queue_usd_replication must only append to REPLICATION_QUEUE — no other side effects."""
    cfg_a = SimpleNamespace(prim_path="/World/envs/env_.*/Robot")
    cfg_b = SimpleNamespace(prim_path="/World/envs/env_.*/Object")

    queue_usd_replication(cfg_a)
    queue_usd_replication(cfg_b)

    assert [(cfg_a, UsdReplicateContext), (cfg_b, UsdReplicateContext)] == REPLICATION_QUEUE


def test_make_clone_plan_homogeneous_returns_env_root_plan(sim):
    """Homogeneous (single-variant) cfgs produce one source row at the env root."""
    cube = SimpleNamespace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
    )

    plan = make_clone_plan(
        cfgs=[cube],
        num_clones=4,
        env_spacing=1.0,
        device=sim.cfg.device,
    )

    assert plan.sources == ("/World/envs/env_0",)
    assert plan.destinations == ("/World/envs/env_{}",)
    assert plan.clone_mask.shape == (1, 4)
    assert plan.clone_mask.all()
    assert plan.cfg_rows[id(cube)] == (0,)
    assert plan.env_ids.shape == (4,)
    assert plan.positions.shape == (4, 3)
    assert cube.spawn.spawn_path == "/World/envs/env_0/Robot"


def test_make_clone_plan_heterogeneous_mutates_spawn_paths(sim):
    """Multi-variant spawners get per-variant spawn_paths and contribute multiple plan rows."""
    multi_cfg = SimpleNamespace(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.ConeCfg(radius=0.1, height=0.2),
                sim_utils.SphereCfg(radius=0.1),
            ]
        ),
    )
    plain_cfg = SimpleNamespace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
    )

    plan = make_clone_plan(
        cfgs=[multi_cfg, plain_cfg],
        num_clones=4,
        env_spacing=1.0,
        device=sim.cfg.device,
        clone_strategy=sequential,
    )

    assert plan.destinations == (
        "/World/envs/env_{}/Object",
        "/World/envs/env_{}/Object",
        "/World/envs/env_{}/Robot",
    )
    assert plan.cfg_rows[id(multi_cfg)] == (0, 1)
    assert plan.cfg_rows[id(plain_cfg)] == (2,)
    assert multi_cfg.spawn.spawn_paths == ["/World/envs/env_0/Object", "/World/envs/env_1/Object"]
    assert plain_cfg.spawn.spawn_path == "/World/envs/env_0/Robot"


def test_make_clone_plan_skips_global_cfgs(sim):
    """Cfgs whose prim_path is not under /World/envs/ are excluded from the plan."""
    global_cfg = SimpleNamespace(
        prim_path="/World/global/Robot",
        spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
    )

    plan = make_clone_plan(
        cfgs=[global_cfg],
        num_clones=3,
        env_spacing=1.0,
        device=sim.cfg.device,
    )

    assert plan.sources == ()
    assert plan.destinations == ()
    assert plan.clone_mask.shape == (0, 3)
    assert plan.cfg_rows == {}


def test_clone_plan_from_env_0_populates_cfg_rows(sim):
    """from_env_0 auto-maps queued env-scoped cfgs to row 0 and excludes global ones."""
    env_cfg_a = SimpleNamespace(prim_path="/World/envs/env_.*/Robot")
    env_cfg_b = SimpleNamespace(prim_path="/World/envs/env_.*/Object")
    global_cfg = SimpleNamespace(prim_path="/World/global/Light")

    queue_usd_replication(env_cfg_a)
    queue_usd_replication(env_cfg_b)
    queue_usd_replication(global_cfg)

    plan = ClonePlan.from_env_0(
        source="/World/envs/env_0",
        destination="/World/envs/env_{}",
        num_clones=4,
        device=sim.cfg.device,
        positions=grid_transforms(4, 1.0, device=sim.cfg.device)[0],
    )

    assert plan.sources == ("/World/envs/env_0",)
    assert plan.destinations == ("/World/envs/env_{}",)
    assert plan.cfg_rows == {id(env_cfg_a): (0,), id(env_cfg_b): (0,)}
    assert plan.clone_mask.all() and plan.clone_mask.shape == (1, 4)
    assert torch.equal(plan.env_ids, torch.arange(4, dtype=torch.long, device=sim.cfg.device))


def test_replicate_drains_queue_dispatches_and_publishes(sim):
    """replicate(plan) drains REPLICATION_QUEUE, calls each backend once, publishes, clears."""

    class FakeCtx:
        replicate_priority = 0
        instances: list["FakeCtx"] = []

        def __init__(self, stage):
            self.stage = stage
            self.queue_calls: list[tuple] = []
            self.replicate_calls = 0
            FakeCtx.instances.append(self)

        def queue_mapping(self, sources, destinations, env_ids, mask, *, positions=None):
            self.queue_calls.append((tuple(sources), tuple(destinations), mask.clone()))

        def replicate(self):
            self.replicate_calls += 1

    cfg_a = SimpleNamespace(prim_path="/World/envs/env_.*/Robot")
    cfg_b = SimpleNamespace(prim_path="/World/envs/env_.*/Object")
    REPLICATION_QUEUE.append((cfg_a, FakeCtx))
    REPLICATION_QUEUE.append((cfg_b, FakeCtx))

    plan = ClonePlan(
        sources=("/World/envs/env_0/Robot", "/World/envs/env_0/Object"),
        destinations=("/World/envs/env_{}/Robot", "/World/envs/env_{}/Object"),
        clone_mask=torch.ones((2, 4), dtype=torch.bool, device=sim.cfg.device),
        env_ids=torch.arange(4, dtype=torch.long, device=sim.cfg.device),
        positions=grid_transforms(4, 1.0, device=sim.cfg.device)[0],
        cfg_rows={id(cfg_a): (0,), id(cfg_b): (1,)},
    )
    sim.set_clone_plan(None)

    replicate(plan, stage=sim_utils.get_current_stage())

    # Exactly one FakeCtx instance is shared across both cfgs, dispatched once per backend
    # with the union of rows the cfgs own.
    assert len(FakeCtx.instances) == 1
    ctx = FakeCtx.instances[0]
    assert len(ctx.queue_calls) == 1
    sources, _destinations, mask = ctx.queue_calls[0]
    assert sources == ("/World/envs/env_0/Robot", "/World/envs/env_0/Object")
    assert mask.shape == (2, 4)
    assert ctx.replicate_calls == 1
    assert sim.get_clone_plan() is plan
    assert REPLICATION_QUEUE == []


def test_replicate_dedupes_shared_rows_across_cfgs(sim):
    """Regression: multiple cfgs sharing the same row dispatch one mapping row, not N.

    In a homogeneous plan every cfg under the env root maps to row 0; without dedup, each
    cfg would tell the backend to re-instantiate row 0 once more, multiplying the count
    of articulations/rigids per world by the number of cfgs.
    """

    class FakeCtx:
        replicate_priority = 0
        instances: list["FakeCtx"] = []

        def __init__(self, stage):
            self.queue_calls: list[tuple] = []
            self.replicate_calls = 0
            FakeCtx.instances.append(self)

        def queue_mapping(self, sources, destinations, env_ids, mask, *, positions=None):
            self.queue_calls.append((tuple(sources), tuple(destinations), mask.clone()))

        def replicate(self):
            self.replicate_calls += 1

    cfgs = [SimpleNamespace(prim_path=f"/World/envs/env_.*/asset_{i}") for i in range(5)]
    for cfg in cfgs:
        REPLICATION_QUEUE.append((cfg, FakeCtx))

    plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=torch.ones((1, 3), dtype=torch.bool, device=sim.cfg.device),
        env_ids=torch.arange(3, dtype=torch.long, device=sim.cfg.device),
        positions=grid_transforms(3, 1.0, device=sim.cfg.device)[0],
        cfg_rows={id(cfg): (0,) for cfg in cfgs},
    )

    replicate(plan, stage=sim_utils.get_current_stage())

    assert len(FakeCtx.instances) == 1
    ctx = FakeCtx.instances[0]
    assert len(ctx.queue_calls) == 1, "shared rows should collapse to one queue_mapping per backend"
    sources, _destinations, mask = ctx.queue_calls[0]
    assert sources == ("/World/envs/env_0",)
    assert mask.shape == (1, 3)
    assert ctx.replicate_calls == 1


def test_replicate_runs_lower_priority_backends_first(sim):
    """Sort order: lower replicate_priority runs first (physics before USD)."""

    call_order: list[str] = []

    class LowPriority:
        replicate_priority = 0

        def __init__(self, stage):
            pass

        def queue_mapping(self, *args, **kwargs):
            pass

        def replicate(self):
            call_order.append("low")

    class HighPriority:
        replicate_priority = 100

        def __init__(self, stage):
            pass

        def queue_mapping(self, *args, **kwargs):
            pass

        def replicate(self):
            call_order.append("high")

    cfg = SimpleNamespace(prim_path="/World/envs/env_.*/Robot")
    REPLICATION_QUEUE.append((cfg, HighPriority))
    REPLICATION_QUEUE.append((cfg, LowPriority))

    plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=torch.ones((1, 2), dtype=torch.bool, device=sim.cfg.device),
        env_ids=torch.arange(2, dtype=torch.long, device=sim.cfg.device),
        positions=None,
        cfg_rows={id(cfg): (0,)},
    )
    replicate(plan, stage=sim_utils.get_current_stage())

    assert call_order == ["low", "high"]


def test_replicate_skips_cfgs_not_in_plan(sim):
    """Cfgs absent from plan.cfg_rows are silently skipped."""
    sentinel = MagicMock()
    sentinel.replicate_priority = 0
    sentinel.replicate.side_effect = lambda: None
    sentinel_cls = MagicMock(return_value=sentinel)

    excluded_cfg = SimpleNamespace(prim_path="/World/global/Skip")
    REPLICATION_QUEUE.append((excluded_cfg, sentinel_cls))

    plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=torch.ones((1, 2), dtype=torch.bool, device=sim.cfg.device),
        env_ids=torch.arange(2, dtype=torch.long, device=sim.cfg.device),
        positions=None,
        cfg_rows={},
    )
    replicate(plan, stage=sim_utils.get_current_stage())

    sentinel_cls.assert_not_called()


def test_replicate_clears_queue_on_backend_failure(sim):
    """REPLICATION_QUEUE is drained even when a backend ctx raises mid-dispatch."""

    class ExplodingCtx:
        replicate_priority = 0

        def __init__(self, stage):
            pass

        def queue_mapping(self, *args, **kwargs):
            pass

        def replicate(self):
            raise RuntimeError("backend boom")

    cfg = SimpleNamespace(prim_path="/World/envs/env_.*/Robot")
    REPLICATION_QUEUE.append((cfg, ExplodingCtx))

    plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=torch.ones((1, 2), dtype=torch.bool, device=sim.cfg.device),
        env_ids=torch.arange(2, dtype=torch.long, device=sim.cfg.device),
        positions=None,
        cfg_rows={id(cfg): (0,)},
    )

    with pytest.raises(RuntimeError, match="backend boom"):
        replicate(plan, stage=sim_utils.get_current_stage())

    assert REPLICATION_QUEUE == []


def test_replicate_session_clears_queue_when_asset_init_fails(sim):
    """ReplicateSession.__exit__ drops queued cfgs if the asset constructor body raises."""
    from isaaclab.cloner import ReplicateSession

    leaked_cfg = SimpleNamespace(prim_path="/World/envs/env_.*/Robot")

    sentinel = MagicMock()
    sentinel_cls = MagicMock(return_value=sentinel)

    with pytest.raises(RuntimeError, match="asset boom"):
        with ReplicateSession(
            cfgs=[],
            num_clones=2,
            env_spacing=1.0,
            device=sim.cfg.device,
            stage=sim_utils.get_current_stage(),
        ):
            REPLICATION_QUEUE.append((leaked_cfg, sentinel_cls))
            raise RuntimeError("asset boom")

    assert REPLICATION_QUEUE == []
    sentinel_cls.assert_not_called()


def test_iter_clone_plan_matches(sim):
    """ClonePlan entries can be matched by destination path expression."""
    plan = ClonePlan(
        sources=("/World/envs/env_0/Object", "/World/envs/env_1/Object"),
        destinations=("/World/envs/env_{}/Object", "/World/envs/env_{}/Object"),
        clone_mask=torch.tensor(
            [[True, False, True, False], [False, True, False, True]],
            dtype=torch.bool,
            device=sim.cfg.device,
        ),
    )

    matches = list(iter_clone_plan_matches(plan, "/World/envs/env_.*/Object/Body/Camera"))

    assert matches == [
        (
            "/World/envs/env_0/Object",
            "/World/envs/env_{}/Object",
            "/World/envs/env_0/Object/Body/Camera",
            (0, 2),
        ),
        (
            "/World/envs/env_1/Object",
            "/World/envs/env_{}/Object",
            "/World/envs/env_1/Object/Body/Camera",
            (1, 3),
        ),
    ]

    plan = ClonePlan(
        sources=("/World/envs/env_3/Object",),
        destinations=("/World/envs/env_{}/Object",),
        clone_mask=torch.tensor([[False, False, True, True]], device=sim.cfg.device),
    )

    matches = list(iter_clone_plan_matches(plan, "/World/envs/env_.*/Object/Body/Camera"))

    assert matches == [
        (
            "/World/envs/env_3/Object",
            "/World/envs/env_{}/Object",
            "/World/envs/env_3/Object/Body/Camera",
            (2, 3),
        )
    ]

    plan = ClonePlan(
        sources=("/World/source/Object",),
        destinations=("/World/scenes/{}/Object",),
        clone_mask=torch.tensor([[True, True]], device=sim.cfg.device),
    )

    matches = list(iter_clone_plan_matches(plan, "/World/scenes/.*/Object/Body/Camera"))

    assert matches == [
        (
            "/World/source/Object",
            "/World/scenes/{}/Object",
            "/World/source/Object/Body/Camera",
            (0, 1),
        )
    ]

    plan = ClonePlan(
        sources=("/World/source",),
        destinations=("/World/scenes/{}",),
        clone_mask=torch.tensor([[True, True]], device=sim.cfg.device),
    )

    matches = list(iter_clone_plan_matches(plan, "/World/scenes/.*/Object/Body/Camera"))

    assert matches == [
        (
            "/World/source",
            "/World/scenes/{}",
            "/World/source/Object/Body/Camera",
            (0, 1),
        )
    ]

    plan = ClonePlan(
        sources=("/World/envs/env_0", "/World/envs/env_0/Object"),
        destinations=("/World/envs/env_{}", "/World/envs/env_{}/Object"),
        clone_mask=torch.tensor([[True, True], [True, True]], device=sim.cfg.device),
    )

    matches = list(iter_clone_plan_matches(plan, "/World/envs/env_.*/Object/Body/Camera"))

    assert matches == [
        (
            "/World/envs/env_0/Object",
            "/World/envs/env_{}/Object",
            "/World/envs/env_0/Object/Body/Camera",
            (0, 1),
        )
    ]


def test_resolve_clone_plan_source_nested_templates_pick_most_specific(sim):
    """A path owned by both an ancestor and a descendant template resolves to the descendant."""
    plan = ClonePlan(
        sources=("/World/envs/env_0/Robot", "/World/envs/env_0/Robot/ee_link/palm_link/Camera"),
        destinations=("/World/envs/env_{}/Robot", "/World/envs/env_{}/Robot/ee_link/palm_link/Camera"),
        clone_mask=torch.tensor([[True, True], [True, True]], device=sim.cfg.device),
    )

    # The camera path matches both templates; the more specific (longer-matching) one wins.
    resolved = resolve_clone_plan_source(plan=plan, path_expr="/World/envs/env_0/Robot/ee_link/palm_link/Camera")

    assert resolved == (
        "/World/envs/env_0/Robot/ee_link/palm_link/Camera",
        "/World/envs/env_*/Robot/ee_link/palm_link/Camera",
        "",
    )

    # A path that only the ancestor template owns still resolves against it with its suffix.
    resolved = resolve_clone_plan_source(plan=plan, path_expr="/World/envs/env_0/Robot/base")

    assert resolved == ("/World/envs/env_0/Robot", "/World/envs/env_*/Robot", "/base")


def test_resolve_clone_plan_source_ambiguous_templates_raise(sim):
    """Two distinct, equally specific templates owning a path remain a genuine ambiguity."""
    plan = ClonePlan(
        sources=("/World/envs/env_0/Robot", "/World/envs/env_0/Robot"),
        destinations=("/World/envs/{}/Robot", "/World/{}/env_0/Robot"),
        clone_mask=torch.tensor([[True, True], [True, True]], device=sim.cfg.device),
    )

    with pytest.raises(ValueError, match="matches multiple destination templates"):
        resolve_clone_plan_source(plan=plan, path_expr="/World/envs/env_0/Robot")


def test_resolve_clone_plan_source_merges_same_template_rows(sim):
    """Heterogeneous source rows sharing one template OR-merge their masks for the coverage check."""
    # One logical asset cloned from two source variants onto the same destination template.
    # Neither row alone covers all envs; row 0 -> envs (0, 2), row 1 -> envs (1, 3).
    plan = ClonePlan(
        sources=("/World/envs/env_0/Object", "/World/envs/env_1/Object"),
        destinations=("/World/envs/env_{}/Object", "/World/envs/env_{}/Object"),
        clone_mask=torch.tensor([[True, False, True, False], [False, True, False, True]], device=sim.cfg.device),
    )

    # The union of both rows covers every env, so resolution succeeds and reports the first row's source.
    resolved = resolve_clone_plan_source(plan=plan, path_expr="/World/envs/env_.*/Object/Body/Camera")

    assert resolved == ("/World/envs/env_0/Object", "/World/envs/env_*/Object", "/Body/Camera")


def test_resolve_clone_plan_source_partial_coverage_raises(sim):
    """When the matching rows' merged mask misses an env, partial coverage is rejected."""
    # Row 0 -> envs (0, 2), row 1 -> env (1); env 3 is covered by neither row.
    plan = ClonePlan(
        sources=("/World/envs/env_0/Object", "/World/envs/env_1/Object"),
        destinations=("/World/envs/env_{}/Object", "/World/envs/env_{}/Object"),
        clone_mask=torch.tensor([[True, False, True, False], [False, True, False, False]], device=sim.cfg.device),
    )

    with pytest.raises(NotImplementedError, match="partial-env heterogeneous coverage"):
        resolve_clone_plan_source(plan=plan, path_expr="/World/envs/env_.*/Object/Body/Camera")
