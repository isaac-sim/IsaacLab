# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest

from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.sim.simulation_context import build_simulation_context

pytestmark = pytest.mark.integration


# each argument maps onto one cfg field, so every value is covered once with the others rotated
@pytest.mark.parametrize(("gravity_enabled", "device", "dt"), [(True, "cpu", 0.01), (False, "cuda:0", 0.1)])
@pytest.mark.isaacsim_ci
def test_build_simulation_context_no_cfg(gravity_enabled, device, dt):
    """Test that the simulation context is built when no simulation cfg is passed in."""
    with build_simulation_context(gravity_enabled=gravity_enabled, device=device, dt=dt) as sim:
        if gravity_enabled:
            assert sim.cfg.gravity == (0.0, 0.0, -9.81)
        else:
            assert sim.cfg.gravity == (0.0, 0.0, 0.0)

        assert sim.cfg.device == device
        assert sim.cfg.dt == dt

        # Ensure that dome light didn't get added automatically as we are headless
        assert not sim.stage.GetPrimAtPath("/World/defaultDomeLight").IsValid()


@pytest.mark.parametrize("add_ground_plane", [True, False])
@pytest.mark.isaacsim_ci
def test_build_simulation_context_ground_plane(add_ground_plane):
    """Test that the simulation context is built with the correct ground plane."""
    with build_simulation_context(add_ground_plane=add_ground_plane) as sim:
        # Ensure that ground plane got added
        if add_ground_plane:
            assert sim.stage.GetPrimAtPath("/World/defaultGroundPlane").IsValid()
        else:
            assert not sim.stage.GetPrimAtPath("/World/defaultGroundPlane").IsValid()


@pytest.mark.parametrize("add_lighting", [True, False])
@pytest.mark.isaacsim_ci
def test_build_simulation_context_auto_add_lighting(add_lighting):
    """Test that the simulation context is built with the correct lighting.

    ``auto_add_lighting`` only adds a light when a GUI is present, so headless runs follow ``add_lighting``.
    """
    with build_simulation_context(add_lighting=add_lighting, auto_add_lighting=True) as sim:
        if add_lighting:
            # Ensure that dome light got added
            assert sim.stage.GetPrimAtPath("/World/defaultDomeLight").IsValid()
        else:
            # Ensure that dome light didn't get added as there's no GUI
            assert not sim.stage.GetPrimAtPath("/World/defaultDomeLight").IsValid()


@pytest.mark.isaacsim_ci
def test_build_simulation_context_cfg():
    """Test that the simulation context honors sim_cfg's values, with an explicit
    device override winning when both ``sim_cfg`` and ``device`` are passed.

    Most test callers pass both kwargs together expecting the device kwarg to
    win; the override branch in :func:`build_simulation_context` exists for
    that case. ``gravity`` and ``dt`` are not overridable by the helper's
    kwargs (only sim_cfg's values are used).
    """
    dt = 0.001
    # Non-standard gravity
    gravity = (0.0, 0.0, -1.81)
    device = "cuda:0"

    cfg = SimulationCfg(
        gravity=gravity,
        device=device,
        dt=dt,
    )

    # Pass only sim_cfg: gravity, device, dt all come from sim_cfg (kwargs ignored).
    with build_simulation_context(sim_cfg=cfg, gravity_enabled=False, dt=0.01) as sim:
        assert sim.cfg.gravity == gravity
        assert sim.cfg.device == device
        assert sim.cfg.dt == dt

    # Pass sim_cfg and an explicit device override: device kwarg wins.
    with build_simulation_context(sim_cfg=cfg, device="cpu") as sim:
        assert sim.cfg.gravity == gravity
        assert sim.cfg.device == "cpu"
        assert sim.cfg.dt == dt
