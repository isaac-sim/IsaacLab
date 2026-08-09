# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualizer correctness on the canonical task-free rendering scene."""

import sys
from pathlib import Path

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import pytest  # noqa: E402

_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

from visualizer_runner import run_visualizer_case  # noqa: E402

pytestmark = [pytest.mark.isaacsim_ci, pytest.mark.cold_cache]

_CASES = [
    pytest.param(physics, visualizer, tiled, id=f"{physics}-{visualizer}-{'tiled' if tiled else 'viewport'}")
    for physics in ("physx", "newton")
    for visualizer in ("kit", "newton")
    for tiled in (False, True)
]


@pytest.mark.parametrize("physics,visualizer,tiled", _CASES)
def test_visualizer_rendering_scene(physics: str, visualizer: str, tiled: bool, request: pytest.FixtureRequest) -> None:
    """Every visualizer mode renders the same reset scene and observes its motion."""
    run_visualizer_case(physics, visualizer, tiled, request)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
