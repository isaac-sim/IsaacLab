# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for physics-manager CUDA device initialization."""

from types import SimpleNamespace

from isaaclab.physics import PhysicsManager
from isaaclab.physics import physics_manager as physics_manager_module


def test_initialize_synchronizes_cuda_before_backend_setup(monkeypatch):
    """The base manager must synchronize CUDA before backend-specific setup."""
    calls = []
    monkeypatch.setattr(physics_manager_module, "set_cuda_device", lambda device: calls.append(("cuda", device)))
    monkeypatch.setattr(PhysicsManager, "_sim", None)
    monkeypatch.setattr(PhysicsManager, "_cfg", None)
    monkeypatch.setattr(PhysicsManager, "_device", "cuda:0")

    class _TestManager(PhysicsManager):
        @classmethod
        def initialize(cls, sim_context):
            super().initialize(sim_context)
            calls.append(("backend", PhysicsManager._device))

    sim_context = SimpleNamespace(cfg=SimpleNamespace(physics=object(), device="cuda:2"))

    _TestManager.initialize(sim_context)

    assert calls == [("cuda", "cuda:2"), ("backend", "cuda:2")]


def test_initialize_does_not_synchronize_cpu_device(monkeypatch):
    """CPU simulation must not initialize or select CUDA runtimes."""
    devices = []
    monkeypatch.setattr(physics_manager_module, "set_cuda_device", devices.append)
    monkeypatch.setattr(PhysicsManager, "_sim", None)
    monkeypatch.setattr(PhysicsManager, "_cfg", None)
    monkeypatch.setattr(PhysicsManager, "_device", "cuda:0")
    sim_context = SimpleNamespace(cfg=SimpleNamespace(physics=object(), device="cpu"))

    PhysicsManager.initialize(sim_context)

    assert devices == []
