# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This test has a lot of duplication with ``test_build_simulation_context_nonheadless.py``. This is intentional to ensure that the
tests are run in both headless and non-headless modes, and we currently can't re-build the simulation app in a script.

If you need to make a change to this test, please make sure to also make the same change to ``test_build_simulation_context_nonheadless.py``.

"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest
from isaacsim.core.utils.prims import is_prim_path_valid

from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.sim.simulation_context import build_simulation_context


@pytest.mark.parametrize("gravity_enabled", [True, False])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("dt", [0.01, 0.1])
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
        assert not is_prim_path_valid("/World/defaultDomeLight")


@pytest.mark.parametrize("add_ground_plane", [True, False])
def test_build_simulation_context_ground_plane(add_ground_plane):
    """Test that the simulation context is built with the correct ground plane."""
    with build_simulation_context(add_ground_plane=add_ground_plane) as _:
        # Ensure that ground plane got added
        assert is_prim_path_valid("/World/defaultGroundPlane") == add_ground_plane


@pytest.mark.parametrize("add_lighting", [True, False])
@pytest.mark.parametrize("auto_add_lighting", [True, False])
def test_build_simulation_context_auto_add_lighting(add_lighting, auto_add_lighting):
    """Test that the simulation context is built with the correct lighting."""
    with build_simulation_context(add_lighting=add_lighting, auto_add_lighting=auto_add_lighting) as _:
        if add_lighting:
            # Ensure that dome light got added
            assert is_prim_path_valid("/World/defaultDomeLight")
        else:
            # Ensure that dome light didn't get added as there's no GUI
            assert not is_prim_path_valid("/World/defaultDomeLight")


def test_build_simulation_context_cfg():
    """Test that the simulation context is built with the correct cfg and values don't get overridden."""
    dt = 0.001
    # Non-standard gravity
    gravity = (0.0, 0.0, -1.81)
    device = "cuda:0"

    cfg = SimulationCfg(
        gravity=gravity,
        device=device,
        dt=dt,
    )

    with build_simulation_context(sim_cfg=cfg, gravity_enabled=False, dt=0.01, device="cpu") as sim:
        assert sim.cfg.gravity == gravity
        assert sim.cfg.device == device
        assert sim.cfg.dt == dt
