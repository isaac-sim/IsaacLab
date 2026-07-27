# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rendering correctness tests for Shadow Hand with PhysX and Isaac Sim RTX."""

# Launch Isaac Sim Simulator first for kit-based combinations.
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
from rendering_test_utils import (  # noqa: E402
    PHYSX_ISAACSIM_RTX_AOV_COMBINATIONS,
    arm_timeout_traceback_watchdog,
    make_attach_comparison_properties_fixture,
    make_determinism_fixture,
    make_generate_html_report_fixture,
    rendering_test_shadow_hand,
)

pytestmark = pytest.mark.isaacsim_ci
_COMPARISON_SCORES: list[dict] = []

_determinism_fixture = make_determinism_fixture()
_generate_html_report_fixture = make_generate_html_report_fixture(_COMPARISON_SCORES, Path(__file__).stem + ".html")
_attach_comparison_properties_fixture = make_attach_comparison_properties_fixture(_COMPARISON_SCORES)

arm_timeout_traceback_watchdog()


@pytest.mark.parametrize("physics_backend,renderer,data_type", PHYSX_ISAACSIM_RTX_AOV_COMBINATIONS)
def test_rendering_shadow_hand_physx_isaacsim_rtx(physics_backend, renderer, data_type):
    """Test Shadow Hand rendering with PhysX and Isaac Sim RTX."""
    rendering_test_shadow_hand(physics_backend, renderer, data_type, _COMPARISON_SCORES)
