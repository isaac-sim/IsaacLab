# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for FabricStageCache lifecycle."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import pytest  # noqa: E402
from isaaclab_physx.sim.fabric_stage_cache import FabricStageCache  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

pytestmark = pytest.mark.isaacsim_ci


@pytest.fixture(autouse=True)
def setup_teardown():
    """Create and clear stage for each test."""
    sim_utils.create_new_stage()
    sim_utils.update_stage()
    yield
    if SimulationContext.instance() is not None:
        sim_utils.clear_stage()
        SimulationContext.clear_instance()


@pytest.fixture()
def cache():
    """Provide a FabricStageCache attached to the simulation stage."""
    sim = SimulationContext()
    return FabricStageCache(sim.stage)


def test_stage_attached(cache):
    """The cached usdrt stage is not None after construction."""
    assert cache.stage is not None


def test_get_hierarchy_returns_handle(cache):
    """get_hierarchy returns a hierarchy handle and fabric id."""
    hierarchy, fabric_id_int = cache.get_hierarchy()
    assert hierarchy is not None
    assert isinstance(fabric_id_int, int)


def test_get_hierarchy_is_cached(cache):
    """Repeated calls return the same hierarchy handle."""
    h1, id1 = cache.get_hierarchy()
    h2, id2 = cache.get_hierarchy()
    assert h1 is h2
    assert id1 == id2


def test_close_clears_state(cache):
    """close() clears internal caches."""
    cache.get_hierarchy()  # populate cache
    cache.close()
    assert cache.stage is None
