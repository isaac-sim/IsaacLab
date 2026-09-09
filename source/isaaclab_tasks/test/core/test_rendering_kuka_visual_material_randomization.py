# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac RTX regression for per-link Kuka visual-material randomization."""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
from rendering_test_utils import (  # noqa: E402
    make_attach_comparison_properties_fixture,
    make_determinism_fixture,
    make_generate_html_report_fixture,
    rendering_test_kuka_visual_material_randomization,
)

pytestmark = [pytest.mark.isaacsim_ci, pytest.mark.integration, pytest.mark.rendering]

_COMPARISON_SCORES: list[dict] = []

_determinism_fixture = make_determinism_fixture()
_generate_html_report_fixture = make_generate_html_report_fixture(_COMPARISON_SCORES, Path(__file__).stem + ".html")
_attach_comparison_properties_fixture = make_attach_comparison_properties_fixture(_COMPARISON_SCORES)


@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_rendering_kuka_visual_material_randomization():
    """Fabric writes must reach Isaac RTX without changing USD."""
    rendering_test_kuka_visual_material_randomization("physx", "isaacsim_rtx_renderer", _COMPARISON_SCORES)
