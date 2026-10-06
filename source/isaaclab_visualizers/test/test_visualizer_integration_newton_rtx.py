# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cartpole env + Newton RTX visualizer frame checks on Newton MJWarp.

Kept apart from ``test_visualizer_integration_newton``: OVRTX cannot share a process with Kit, which
that module launches at import.
"""

import sys
from pathlib import Path

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(NewtonCfg(solver_cfg=MJWarpSolverCfg()), visualizer=["newton_rtx"])


import pytest  # noqa: E402

_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

import visualizer_integration_utils as _viz_utils  # noqa: E402

pytestmark = [pytest.mark.arm_ci]


def test_cartpole_env_newton_rtx_visualizer_motion_with_play_pause_newton(caplog: pytest.LogCaptureFixture) -> None:
    """Newton RTX frames are non-flat, move while playing, freeze while paused, and draw a marker."""
    _viz_utils.run_cartpole_env_visualizers_motion_with_play_pause("newton", caplog, visualizer_kinds=("newton_rtx",))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
