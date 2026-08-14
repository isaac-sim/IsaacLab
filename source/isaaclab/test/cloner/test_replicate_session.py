# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for clone-context dispatch without a simulator runtime."""

from types import SimpleNamespace

import pytest
import torch

import isaaclab.cloner.replicate_session as replicate_session
from isaaclab.cloner import ClonePlan
from isaaclab.physics import PhysicsEvent
from isaaclab.sensors import SensorBaseCfg
from isaaclab.sim import SimulationContext


def test_sensor_default_does_not_request_a_cloning_context():
    """Sensors rely on automatic Kit replication unless a user explicitly overrides it."""

    assert SensorBaseCfg().cloning_contexts == ()


@pytest.mark.parametrize(
    ("clone_mask", "source_ids", "expected_prototypes"),
    [
        (torch.zeros((0, 4), dtype=torch.bool), (), {0}),
        (
            torch.tensor(
                [
                    [False, True, False, False],
                    [False, False, False, True],
                ]
            ),
            (1, 3),
            {0, 1, 3},
        ),
        (
            torch.tensor(
                [
                    [False, True, False, False],
                    [False, False, False, False],
                ]
            ),
            (1, 7),
            {0, 1},
        ),
    ],
)
def test_replicate_session_positions_only_prototype_roots(monkeypatch, clone_mask, source_ids, expected_prototypes):
    """The session authors active prototype roots without pre-authoring destinations."""
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    positions = torch.tensor(
        [
            [-1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    plan = ClonePlan(
        sources=tuple(f"/World/envs/env_{env_id}/Object" for env_id in source_ids),
        destinations=tuple("/World/envs/env_{}/Object" for _ in source_ids),
        clone_mask=clone_mask,
        env_ids=torch.arange(4),
        positions=positions,
        env_template="/World/envs/env_{}",
    )
    stage = Usd.Stage.CreateInMemory()
    monkeypatch.setattr(replicate_session, "make_clone_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(replicate_session, "replicate", lambda *_args, **_kwargs: None)

    with replicate_session.ReplicateSession([], 4, 2.0, "cpu", stage=stage):
        for env_id in range(4):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}")
            assert prim.IsValid() is (env_id in expected_prototypes)
            if prim.IsValid():
                xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                assert tuple(xform.ExtractTranslation()) == pytest.approx(positions[env_id].tolist())


@pytest.mark.parametrize(
    ("kit_available", "explicit_request", "replicate_physics", "expected_sync", "expected_deferred"),
    [
        (False, False, True, 0, False),
        (False, True, True, 1, False),
        (True, False, True, 0, True),
        (True, True, True, 1, False),
        (True, False, False, 1, False),
    ],
)
def test_replicate_distinguishes_automatic_and_explicit_usd_contexts(
    monkeypatch, kit_available, explicit_request, replicate_physics, expected_sync, expected_deferred
):
    """Only implicit Kit USD cloning with physics waits for MODEL_INIT."""

    class FakeUsdContext:
        replicate_priority = 100
        instances: list["FakeUsdContext"] = []

        def __init__(self, stage):
            self.replicate_calls = 0
            FakeUsdContext.instances.append(self)

        def queue_mapping(self, sources, destinations, env_ids, mask, *, positions=None):
            pass

        def replicate(self, payload=None):
            self.replicate_calls += 1

    class FakePhysicsManager:
        def __init__(self):
            self.registrations = []

        def register_callback(self, callback, event, order=0, wrap_weak_ref=True):
            self.registrations.append((callback, event, order, wrap_weak_ref))
            return object()

    physics_manager = FakePhysicsManager()
    published = SimpleNamespace(plan=None, physics_manager=physics_manager)
    published.set_clone_plan = lambda plan: setattr(published, "plan", plan)
    monkeypatch.setattr(replicate_session, "UsdReplicateContext", FakeUsdContext)
    monkeypatch.setattr(replicate_session, "has_kit", lambda: kit_available)
    monkeypatch.setattr(replicate_session.FactoryBase, "_get_backend", lambda: "newton")
    monkeypatch.setattr(SimulationContext, "instance", lambda: published)

    cfg = SimpleNamespace(
        prim_path="/World/envs/env_[^/]+/Robot",
        cloning_contexts=(FakeUsdContext,) if explicit_request else (),
        spawn=object(),
    )
    replicate_session.REPLICATION_QUEUE.append(cfg)
    plan = ClonePlan(
        sources=("/World/envs/env_0/Robot",),
        destinations=("/World/envs/env_{}/Robot",),
        clone_mask=torch.ones((1, 2), dtype=torch.bool),
        env_ids=torch.arange(2, dtype=torch.long),
        positions=torch.zeros((2, 3)),
        cfg_rows={id(cfg): (0,)},
    )

    replicate_session.replicate(plan, stage=object(), replicate_physics=replicate_physics)

    assert len(FakeUsdContext.instances) == int(expected_sync > 0 or expected_deferred)
    assert published.plan is plan
    if not FakeUsdContext.instances:
        assert physics_manager.registrations == []
        return

    ctx = FakeUsdContext.instances[0]
    assert ctx.replicate_calls == expected_sync
    if expected_deferred:
        callback, event, order, wrap_weak_ref = physics_manager.registrations[0]
        assert event is PhysicsEvent.MODEL_INIT and order == 2 and wrap_weak_ref is False
        callback(None)
        assert ctx.replicate_calls == 1
    else:
        assert physics_manager.registrations == []


def test_explicit_usd_context_owns_a_shared_implicit_row(monkeypatch):
    """A synchronous explicit root row is not copied again by an implicit deferred context."""

    class FakeUsdContext:
        replicate_priority = 100
        instances = []

        def __init__(self, stage):
            self.replicate_calls = 0
            FakeUsdContext.instances.append(self)

        def queue_mapping(self, sources, destinations, env_ids, mask, *, positions=None):
            pass

        def replicate(self, payload=None):
            self.replicate_calls += 1

    registrations = []
    physics_manager = SimpleNamespace(
        register_callback=lambda callback, event, order=0, wrap_weak_ref=True: registrations.append(
            (callback, event, order, wrap_weak_ref)
        )
    )
    published = SimpleNamespace(plan=None, physics_manager=physics_manager)
    published.set_clone_plan = lambda plan: setattr(published, "plan", plan)
    monkeypatch.setattr(replicate_session, "UsdReplicateContext", FakeUsdContext)
    monkeypatch.setattr(replicate_session, "has_kit", lambda: True)
    monkeypatch.setattr(replicate_session.FactoryBase, "_get_backend", lambda: "newton")
    monkeypatch.setattr(SimulationContext, "instance", lambda: published)

    explicit_cfg = SimpleNamespace(cloning_contexts=(FakeUsdContext,), spawn=object())
    implicit_cfg = SimpleNamespace(cloning_contexts=(), spawn=object())
    replicate_session.REPLICATION_QUEUE.extend((explicit_cfg, implicit_cfg))
    plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=torch.ones((1, 2), dtype=torch.bool),
        env_ids=torch.arange(2),
        positions=torch.zeros((2, 3)),
        cfg_rows={id(explicit_cfg): (0,), id(implicit_cfg): (0,)},
    )

    replicate_session.replicate(plan, stage=object())

    assert len(FakeUsdContext.instances) == 1
    assert FakeUsdContext.instances[0].replicate_calls == 1
    assert registrations == []
