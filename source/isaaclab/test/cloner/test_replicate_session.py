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
from isaaclab.sensors import SensorBaseCfg
from isaaclab.sim import SimulationContext


def test_sensor_default_requests_usd_replication():
    """Sensors ask for USD cloning outright rather than relying on the Kit-gated context."""

    assert SensorBaseCfg().cloning_contexts == ("isaaclab.cloner:UsdReplicateContext",)


def test_replicate_clones_a_default_sensor_without_kit(monkeypatch):
    """A stock sensor cfg must author its per-environment prims when Kit is absent.

    Backends resolve sensor views by matching the cfg's path expression against the stage, so a
    camera left only in ``env_0`` fails the per-environment count check at initialization.
    """
    from pxr import Usd

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/envs/env_0", "Xform")
    stage.DefinePrim("/World/envs/env_1", "Xform")
    stage.DefinePrim("/World/envs/env_0/Camera", "Camera")

    published = SimpleNamespace(plan=None)
    published.set_clone_plan = lambda plan: setattr(published, "plan", plan)
    monkeypatch.setattr(replicate_session, "has_kit", lambda: False)
    monkeypatch.setattr(replicate_session.FactoryBase, "_get_backend", lambda: "newton")
    monkeypatch.setattr(SimulationContext, "instance", lambda: published)

    cfg = SensorBaseCfg(prim_path="/World/envs/env_.*/Camera")
    replicate_session.REPLICATION_QUEUE.append(cfg)
    plan = ClonePlan(
        sources=("/World/envs/env_0/Camera",),
        destinations=("/World/envs/env_{}/Camera",),
        clone_mask=torch.ones((1, 2), dtype=torch.bool),
        env_ids=torch.arange(2, dtype=torch.long),
        positions=torch.zeros((2, 3)),
        cfg_rows={id(cfg): (0,)},
    )

    replicate_session.replicate(plan, stage=stage)

    assert stage.GetPrimAtPath("/World/envs/env_1/Camera").IsValid()


@pytest.mark.parametrize(
    ("kit_available", "explicit_request", "expected_instances"),
    [(False, False, 0), (False, True, 1), (True, False, 1)],
)
def test_replicate_distinguishes_automatic_and_explicit_usd_contexts(
    monkeypatch, kit_available, explicit_request, expected_instances
):
    """Kit gates automatic USD cloning without overriding an explicit cfg request."""

    class FakeUsdContext:
        replicate_priority = 100
        instances: list["FakeUsdContext"] = []

        def __init__(self, stage):
            FakeUsdContext.instances.append(self)

        def queue_mapping(self, sources, destinations, env_ids, mask, *, positions=None):
            pass

        def replicate(self):
            pass

    published = SimpleNamespace(plan=None)
    published.set_clone_plan = lambda plan: setattr(published, "plan", plan)
    monkeypatch.setattr(replicate_session, "UsdReplicateContext", FakeUsdContext)
    monkeypatch.setattr(replicate_session, "has_kit", lambda: kit_available)
    monkeypatch.setattr(replicate_session.FactoryBase, "_get_backend", lambda: "newton")
    monkeypatch.setattr(SimulationContext, "instance", lambda: published)

    cfg = SimpleNamespace(
        prim_path="/World/envs/env_.*/Robot",
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

    replicate_session.replicate(plan, stage=object())

    assert len(FakeUsdContext.instances) == expected_instances
    assert published.plan is plan
