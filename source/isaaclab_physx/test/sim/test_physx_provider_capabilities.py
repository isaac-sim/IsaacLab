# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capability registration tests for :class:`PhysxSceneDataProvider`."""

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


def _build_physx_provider(sim, *, requires_newton_model: bool, requires_usd_stage: bool = False):
    """Force a PhysX provider with the requested SceneDataRequirement.

    The sim's :meth:`SimulationContext.get_scene_data_requirements` is
    monkey-patched for the duration of the test.
    """
    from isaaclab_physx.scene_data_providers import PhysxSceneDataProvider

    sim.get_scene_data_requirements = lambda: SceneDataRequirement(
        requires_newton_model=requires_newton_model,
        requires_usd_stage=requires_usd_stage,
    )
    return PhysxSceneDataProvider(sim.stage, sim)


@pytest.fixture
def sim():
    with build_simulation_context(device="cuda:0", dt=0.01, add_lighting=False) as sim:
        yield sim


def test_physx_baseline_capabilities(sim):
    """Without a Newton consumer, PhysX exposes GpuTransformBuffer and UsdFabric only."""
    provider = _build_physx_provider(sim, requires_newton_model=False)
    caps = provider.list_capabilities()
    assert GpuTransformBuffer in caps
    assert UsdFabric in caps
    assert NewtonState not in caps


def test_physx_newton_state_registered_when_consumer_requires(sim):
    """A Newton-flavoured consumer flips on the NewtonState cap."""
    provider = _build_physx_provider(sim, requires_newton_model=True)
    caps = provider.list_capabilities()
    assert NewtonState in caps


def test_physx_gpu_transform_buffer_handle_works(sim):
    """The registered GpuTransformBuffer handle forwards the typed API."""
    provider = _build_physx_provider(sim, requires_newton_model=False)
    cap = provider.get_capability(GpuTransformBuffer)
    assert cap is not None
    assert isinstance(cap, GpuTransformBuffer)
    # Source format should match what the provider reports.
    assert cap.get_source_format() == provider.get_source_format()


def test_physx_usd_fabric_is_no_op(sim):
    """PhysX writes Fabric natively; ensure_current is a fast no-op."""
    provider = _build_physx_provider(sim, requires_newton_model=False)
    cap = provider.get_capability(UsdFabric)
    assert cap is not None
    cap.ensure_current()  # no error, no return


def test_physx_newton_state_handle_returns_state(sim):
    """The NewtonState cap forwards to the provider's synthetic state."""
    provider = _build_physx_provider(sim, requires_newton_model=True)
    cap = provider.get_capability(NewtonState)
    assert cap is not None
    # Identity: the cap must surface the same model object the provider holds.
    assert cap.get_model() is provider.get_newton_model()
    assert cap.get_state() is provider.get_newton_state()
