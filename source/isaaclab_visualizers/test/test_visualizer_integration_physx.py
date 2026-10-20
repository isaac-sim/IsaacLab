# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cartpole env + Kit visualizer integration tests on PhysX.

Viewport and tiled scenes share one Kit process. Other viewer play/pause behavior is
backend-independent and covered by the Newton integration tests.
"""

import sys
from pathlib import Path

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(enable_cameras=True, visualizer="kit")


import pytest  # noqa: E402

_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

import visualizer_integration_utils as _viz_utils  # noqa: E402

pytestmark = [pytest.mark.isaacsim_ci]


def test_cartpole_env_kit_physx(caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]) -> None:
    """Kit RTX viewport frames move, pause, and resume on PhysX."""
    _viz_utils.run_cartpole_env_visualizers_motion_with_play_pause("physx", caplog, visualizer_kinds=("kit",))
    _viz_utils.assert_no_newton_imgui_bundle_warning(capsys, caplog)


@pytest.mark.flaky(max_runs=2, min_passes=1)
def test_visualizer_tiled_integration_physx(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """Cartpole env + tiled Kit/Newton visualizers on PhysX."""
    _viz_utils.run_cartpole_env_visualizers_tiled_camera_motion("physx", caplog)
    _viz_utils.assert_no_newton_imgui_bundle_warning(capsys, caplog)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
