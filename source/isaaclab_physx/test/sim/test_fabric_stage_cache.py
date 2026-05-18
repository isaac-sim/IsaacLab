# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for FabricStageCache service lifecycle."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import pytest  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab_physx.sim.fabric_stage_cache import FabricStageCache  # noqa: E402

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


class TestFabricStageCacheService:
    """Test that FabricStageCache integrates with SimulationContext's service locator."""

    def test_register_and_retrieve(self):
        """Service can be registered and retrieved via services[]."""
        sim_context = SimulationContext()
        cache = FabricStageCache(sim_context.stage)
        sim_context.services[FabricStageCache] = cache

        retrieved = sim_context.services[FabricStageCache]
        assert retrieved is cache

    def test_stage_attached(self):
        """The cached usdrt stage is not None after construction."""
        sim_context = SimulationContext()
        cache = FabricStageCache(sim_context.stage)
        assert cache.stage is not None

    def test_get_hierarchy_returns_handle(self):
        """get_hierarchy returns a hierarchy handle and fabric id."""
        sim_context = SimulationContext()
        cache = FabricStageCache(sim_context.stage)

        hierarchy, fabric_id_int = cache.get_hierarchy()
        assert hierarchy is not None
        assert isinstance(fabric_id_int, int)

    def test_get_hierarchy_is_cached(self):
        """Repeated calls return the same hierarchy handle."""
        sim_context = SimulationContext()
        cache = FabricStageCache(sim_context.stage)

        h1, id1 = cache.get_hierarchy()
        h2, id2 = cache.get_hierarchy()
        assert h1 is h2
        assert id1 == id2

    def test_close_clears_state(self):
        """close() clears internal caches."""
        sim_context = SimulationContext()
        cache = FabricStageCache(sim_context.stage)
        cache.get_hierarchy()  # populate cache
        cache.close()
        assert cache.stage is None

    def test_clear_instance_closes_service(self):
        """SimulationContext.clear_instance() calls close() on registered services."""
        sim_context = SimulationContext()
        cache = FabricStageCache(sim_context.stage)
        sim_context.services[FabricStageCache] = cache

        SimulationContext.clear_instance()

        # After clear_instance, the cache should have been closed
        assert cache.stage is None

    def test_replacement_caller_closes_old(self):
        """Caller is responsible for closing old service before replacing."""
        sim_context = SimulationContext()
        cache1 = FabricStageCache(sim_context.stage)
        sim_context.services[FabricStageCache] = cache1

        cache1.close()
        cache2 = FabricStageCache(sim_context.stage)
        sim_context.services[FabricStageCache] = cache2

        # Old cache was closed by caller
        assert cache1.stage is None
        # New cache should be active
        assert cache2.stage is not None
