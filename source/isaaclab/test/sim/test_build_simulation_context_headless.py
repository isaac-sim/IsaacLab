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

pytestmark = [pytest.mark.integration, pytest.mark.isaacsim_ci]


@pytest.mark.parametrize(
    ("gravity_enabled", "device", "dt"),
    [(True, "cuda:0", 0.01), (False, "cpu", 0.1)],
    ids=["gravity_cuda", "no_gravity_cpu"],
)
def test_build_simulation_context_no_cfg(gravity_enabled, device, dt):
    """Without a cfg, the helper builds one from its keyword arguments and adds nothing to the scene."""
    with build_simulation_context(gravity_enabled=gravity_enabled, device=device, dt=dt) as sim:
        assert sim.cfg.gravity == ((0.0, 0.0, -9.81) if gravity_enabled else (0.0, 0.0, 0.0))
        assert sim.cfg.device == device
        assert sim.cfg.dt == dt
        assert not sim.stage.GetPrimAtPath("/World/defaultGroundPlane").IsValid()
        assert not sim.stage.GetPrimAtPath("/World/defaultDomeLight").IsValid()


@pytest.mark.parametrize(
    ("add_ground_plane", "add_lighting", "auto_add_lighting"),
    [(True, True, False), (False, False, True)],
    ids=["explicit_extras", "auto_lighting_only"],
)
def test_build_simulation_context_scene_extras(add_ground_plane, add_lighting, auto_add_lighting):
    """Ground plane and dome light follow their flags; auto lighting needs a GUI, which headless lacks."""
    with build_simulation_context(
        add_ground_plane=add_ground_plane, add_lighting=add_lighting, auto_add_lighting=auto_add_lighting
    ) as sim:
        assert sim.stage.GetPrimAtPath("/World/defaultGroundPlane").IsValid() == add_ground_plane
        assert sim.stage.GetPrimAtPath("/World/defaultDomeLight").IsValid() == add_lighting


def test_build_simulation_context_cfg():
    """The helper honors ``sim_cfg``; an explicit ``device`` overrides ``sim_cfg.device``.

    Most test callers pass both kwargs together expecting the device kwarg to win; the override
    branch in :func:`build_simulation_context` exists for that case. ``gravity`` and ``dt`` are
    not overridable by the helper's kwargs (only sim_cfg's values are used).
    """
    cfg = SimulationCfg(gravity=(0.0, 0.0, -1.81), device="cuda:0", dt=0.001)

    # Pass only sim_cfg: gravity, device, dt all come from sim_cfg (kwargs ignored).
    with build_simulation_context(sim_cfg=cfg, gravity_enabled=False, dt=0.01) as sim:
        assert (sim.cfg.gravity, sim.cfg.device, sim.cfg.dt) == ((0.0, 0.0, -1.81), "cuda:0", 0.001)

    # Pass sim_cfg and an explicit device override: device kwarg wins.
    with build_simulation_context(sim_cfg=cfg, device="cpu") as sim:
        assert (sim.cfg.gravity, sim.cfg.device, sim.cfg.dt) == ((0.0, 0.0, -1.81), "cpu", 0.001)
