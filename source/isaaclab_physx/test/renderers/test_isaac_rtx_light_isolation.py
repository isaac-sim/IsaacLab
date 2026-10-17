# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rendered light isolation through the Isaac RTX USD scene."""

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(enable_cameras=True)

import sys
from pathlib import Path

import pytest
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererGlobalSettingsCfg

from isaaclab.sim import build_simulation_context
from isaaclab.test.utils import DeviceScope, test_devices

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "isaaclab" / "test" / "renderers"))
from light_rendering_contract import run_light_isolation_contract  # noqa: E402


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_per_environment_analytic_light_resets(device: str, tmp_path: Path) -> None:
    """Selected analytic light resets preserve the other environments' sensor views."""
    cfg = IsaacRtxRendererCfg(global_settings=IsaacRtxRendererGlobalSettingsCfg(show_all_partitions_by_default=False))
    with build_simulation_context(device=device) as sim:
        # Isaac Sim 6.2 still shares dome backgrounds across partitioned cameras.
        run_light_isolation_contract(sim, cfg, tmp_path, per_environment_domes=False, env_spacing=4.0)
