# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared fixtures for the manager tests.

The managers only read ``num_envs``, ``device`` and ``dt`` from the environment and check ``sim.is_playing()``
to resolve their terms, so a tiny environment double lets the tests run without launching the simulator.
"""

from __future__ import annotations

from typing import Any

import pytest


class PlayingSim:
    """Simulation double that reports a playing timeline."""

    def is_playing(self) -> bool:
        return True


class DummyEnv:
    """Minimal environment double for the managers."""

    def __init__(self, num_envs: int = 20, device: str = "cpu", dt: float = 0.01, **attrs: Any) -> None:
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.sim = PlayingSim()
        for name, value in attrs.items():
            setattr(self, name, value)


@pytest.fixture
def make_env() -> type[DummyEnv]:
    """Return the environment double class so tests can attach custom attributes."""
    return DummyEnv


@pytest.fixture
def env() -> DummyEnv:
    """Return a CPU environment double with 20 environments."""
    return DummyEnv()
