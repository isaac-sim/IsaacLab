# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rendered light isolation through both OVRTX scene ownership paths."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("ovrtx")
pytest.importorskip("ovstage")

from isaaclab_newton.physics import NewtonCfg
from isaaclab_ov.renderers import OVRTXRendererCfg

from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.test.utils import DeviceScope, test_devices

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "isaaclab" / "test" / "renderers"))
from light_rendering_contract import run_light_isolation_contract  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.rendering, pytest.mark.isaacsim_ci]


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("use_ovstage", [False, True])
def test_per_environment_light_resets(monkeypatch: pytest.MonkeyPatch, device: str, use_ovstage: bool) -> None:
    """Native clones and detached OVStage copies both preserve independent lighting."""
    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", "1" if use_ovstage else "0")
    cfg = SimulationCfg(device=device, physics=NewtonCfg())
    with build_simulation_context(sim_cfg=cfg) as sim:
        run_light_isolation_contract(sim, OVRTXRendererCfg(show_all_partitions_by_default=False))
