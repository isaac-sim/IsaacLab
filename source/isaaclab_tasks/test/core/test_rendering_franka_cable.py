# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rendering correctness tests for the Franka cable camera setup."""

# Launch Isaac Sim Simulator first for kit-based combinations.
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
from rendering_test_utils import (  # noqa: E402
    PHYSICS_RENDERER_AOV_GROUPS,
    make_attach_comparison_properties_fixture,
    make_determinism_fixture,
    make_generate_html_report_fixture,
    rendering_test_franka_cable,
)

pytestmark = pytest.mark.isaacsim_ci

# Newton cables have no PhysX preset, so only the Newton physics rows apply.
_RENDERING_PARAMS = [param for param in PHYSICS_RENDERER_AOV_GROUPS if param.values[0] != "physx"]
_COMPARISON_SCORES: list[dict] = []

_determinism_fixture = make_determinism_fixture()
_generate_html_report_fixture = make_generate_html_report_fixture(_COMPARISON_SCORES, Path(__file__).stem + ".html")
_attach_comparison_properties_fixture = make_attach_comparison_properties_fixture(_COMPARISON_SCORES)


@pytest.mark.parametrize("physics_backend,renderer,data_types", _RENDERING_PARAMS)
def test_rendering_franka_cable(physics_backend, renderer, data_types):
    """Test Franka cable rendering correctness across AOVs."""
    rendering_test_franka_cable(physics_backend, renderer, data_types, _COMPARISON_SCORES)
