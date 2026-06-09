# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cartpole env + all non-tiled visualizers on Newton MJWarp."""

import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# launch Kit app
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import pytest  # noqa: E402

_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

import visualizer_integration_utils as _viz_utils  # noqa: E402

_viz_utils.set_visualizer_integration_simulation_app(simulation_app)

run_cartpole_env_visualizers_motion_with_play_pause = _viz_utils.run_cartpole_env_visualizers_motion_with_play_pause

pytestmark = [pytest.mark.isaacsim_ci, pytest.mark.flaky(max_runs=2, min_passes=1)]


def test_cartpole_env_visualizers_motion_with_play_pause_newton(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """Cartpole env + all non-tiled visualizers on Newton MJWarp."""
    run_cartpole_env_visualizers_motion_with_play_pause("newton", caplog)
    _viz_utils.assert_no_newton_imgui_bundle_warning(capsys, caplog)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
