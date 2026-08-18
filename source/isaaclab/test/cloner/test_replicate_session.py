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


def test_sensor_default_does_not_request_a_cloning_context():
    """Sensors rely on automatic Kit replication unless a user explicitly overrides it."""

    assert SensorBaseCfg().cloning_contexts == ()


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

    replicate_session.replicate(plan, stage=object())

    assert len(FakeUsdContext.instances) == expected_instances
    assert published.plan is plan
