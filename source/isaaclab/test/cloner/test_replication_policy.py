# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for automatic USD replication policy."""

from types import SimpleNamespace

from isaaclab.cloner.replicate_session import _automatic_usd_replication_required
from isaaclab.sim import SimulationContext


def _sim(**overrides):
    values = {
        "has_gui": False,
        "has_offscreen_render": False,
        "resolve_visualizer_types": lambda: [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_pure_headless_newton_renderer_skips_automatic_usd_replication(monkeypatch):
    """Newton physics plus Newton rendering consumes the clone plan without USD clones."""
    monkeypatch.setattr(SimulationContext, "instance", classmethod(lambda cls: _sim()))
    context = type("NewtonContext", (), {"__module__": "isaaclab_newton.cloner"})
    camera_cfg = SimpleNamespace(renderer_cfg=SimpleNamespace(renderer_type="newton_warp"))

    assert not _automatic_usd_replication_required([camera_cfg], context)


def test_non_newton_renderer_keeps_automatic_usd_replication(monkeypatch):
    """RTX still receives concrete USD destinations when Newton provides physics."""
    monkeypatch.setattr(SimulationContext, "instance", classmethod(lambda cls: _sim()))
    context = type("NewtonContext", (), {"__module__": "isaaclab_newton.cloner"})
    camera_cfg = SimpleNamespace(renderer_cfg=SimpleNamespace(renderer_type="isaac_rtx"))

    assert _automatic_usd_replication_required([camera_cfg], context)


def test_visualizer_keeps_automatic_usd_replication(monkeypatch):
    """A visualizer remains a USD clone consumer even with Newton Warp cameras."""
    sim = _sim(resolve_visualizer_types=lambda: ["kit"])
    monkeypatch.setattr(SimulationContext, "instance", classmethod(lambda cls: sim))
    context = type("NewtonContext", (), {"__module__": "isaaclab_newton.cloner"})
    camera_cfg = SimpleNamespace(renderer_cfg=SimpleNamespace(renderer_type="newton_warp"))

    assert _automatic_usd_replication_required([camera_cfg], context)
