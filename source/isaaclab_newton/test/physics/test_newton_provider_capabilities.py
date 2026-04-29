# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capability registration tests for :class:`NewtonSceneDataProvider`."""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest
import warp as wp
from isaaclab_newton.physics.capabilities import NewtonState

from isaaclab.physics import GpuTransformBuffer, UsdFabric
from isaaclab.physics.scene_data_requirements import SceneDataRequirement
from isaaclab.sim import build_simulation_context

wp.init()

pytestmark = pytest.mark.isaacsim_ci


def _build_newton_provider(sim, *, requires_usd_stage: bool = False):
    from isaaclab_newton.scene_data_providers import NewtonSceneDataProvider

    sim.get_scene_data_requirements = lambda: SceneDataRequirement(
        requires_usd_stage=requires_usd_stage,
    )
    return NewtonSceneDataProvider(sim.stage, sim)


@pytest.fixture
def sim():
    # Newton backend selected via build_simulation_context.
    with build_simulation_context(device="cuda:0", dt=0.01, add_lighting=False, sim_cfg=None) as sim:
        yield sim


def test_newton_baseline_capabilities(sim):
    """Without a USD consumer, Newton exposes GpuTransformBuffer and NewtonState only."""
    provider = _build_newton_provider(sim, requires_usd_stage=False)
    caps = provider.list_capabilities()
    assert GpuTransformBuffer in caps
    assert NewtonState in caps
    assert UsdFabric not in caps


def test_newton_usd_fabric_registered_when_consumer_requires(sim):
    """A USD-Fabric consumer flips on the UsdFabric cap on Newton."""
    provider = _build_newton_provider(sim, requires_usd_stage=True)
    caps = provider.list_capabilities()
    assert UsdFabric in caps


def test_newton_gpu_transform_buffer_handle_works(sim):
    provider = _build_newton_provider(sim, requires_usd_stage=False)
    cap = provider.get_capability(GpuTransformBuffer)
    assert cap is not None
    assert isinstance(cap, GpuTransformBuffer)
    assert cap.get_source_format() == provider.get_source_format()


def test_newton_state_handle_returns_state(sim):
    provider = _build_newton_provider(sim, requires_usd_stage=False)
    cap = provider.get_capability(NewtonState)
    assert cap is not None
    assert cap.get_model() is provider.get_newton_model()
    assert cap.get_state() is provider.get_newton_state()


def test_newton_usd_fabric_runs_sync(sim, monkeypatch):
    """When USD is required, ensure_current invokes the manager's sync."""
    provider = _build_newton_provider(sim, requires_usd_stage=True)
    cap = provider.get_capability(UsdFabric)
    assert cap is not None

    sync_calls = {"count": 0}
    from isaaclab_newton.physics import NewtonManager

    monkeypatch.setattr(
        NewtonManager,
        "sync_transforms_to_usd",
        lambda *a, **kw: sync_calls.__setitem__("count", sync_calls["count"] + 1),
    )

    cap.ensure_current()
    assert sync_calls["count"] == 1
