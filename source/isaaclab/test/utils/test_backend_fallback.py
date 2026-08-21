# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from isaaclab.sim.simulation_context import SimulationContext
from isaaclab.utils.backend_utils import FactoryBase

pytestmark = pytest.mark.unit


def test_factory_backend_falls_back_to_newton_without_simulation_context(monkeypatch):
    monkeypatch.setattr(SimulationContext, "_instance", None)

    assert FactoryBase._get_backend() == "newton"
