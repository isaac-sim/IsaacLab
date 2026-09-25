# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Golden render test: Shadow Hand with yellow camera background (RGB only)."""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import pytest  # noqa: E402
from rendering_test_utils import rendering_test_shadow_hand_yellow_bg  # noqa: E402

pytestmark = pytest.mark.isaacsim_ci


@pytest.mark.parametrize(
    "physics_backend,renderer",
    [
        ("physx", "isaacsim_rtx_renderer"),
        ("newton", "isaacsim_rtx_renderer"),
        ("physx", "newton_renderer"),
    ],
)
def test_rgb_yellow_background(physics_backend, renderer):
    """Golden render test: RGB output with yellow background."""
    rendering_test_shadow_hand_yellow_bg(physics_backend, renderer, comparison_scores=[])
