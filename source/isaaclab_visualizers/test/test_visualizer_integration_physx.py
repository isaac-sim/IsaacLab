# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cartpole env + Kit visualizer integration tests on PhysX.

Tests focus on Kit RTX rendering with the PhysX backend — the behavior that is unique to this
combination.  Newton viewer, Rerun, and Viser play/pause behavior is backend-agnostic and is
already covered by the Newton integration test (which runs in ~50 s vs ~8 min on PhysX).
"""

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

pytestmark = [pytest.mark.isaacsim_ci]


def test_cartpole_env_kit_physx(caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]) -> None:
    """Kit RTX viewport + tiled camera motion tests on PhysX in one env.

    Runs both the viewport and tiled motion tests in a single env to avoid creating and
    destroying two separate envs.  PhysX env setup/teardown (~3–4 min each) dominates the
    total runtime; sharing one env cuts the expected time roughly in half.

    Newton viewer, Rerun, and Viser are backend-agnostic and are covered by the Newton
    integration test which runs in ~50 s total.
    """
    _viz_utils.run_cartpole_env_kit_viewport_and_tiled("physx", caplog)
    _viz_utils.assert_no_newton_imgui_bundle_warning(capsys, caplog)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
