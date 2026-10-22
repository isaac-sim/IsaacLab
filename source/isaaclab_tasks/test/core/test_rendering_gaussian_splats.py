# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rendering correctness tests for 3D Gaussian scenes and objects with the Kit-based renderers."""

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(enable_cameras=True)

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
from rendering_test_utils import (  # noqa: E402
    GAUSSIAN_SPLATS_KIT_RENDERING_PARAMS,
    make_attach_comparison_properties_fixture,
    make_determinism_fixture,
    make_generate_html_report_fixture,
    rendering_test_gaussian_splats,
)

pytestmark = pytest.mark.isaacsim_ci

_COMPARISON_SCORES: list[dict] = []

_determinism_fixture = make_determinism_fixture()
_generate_html_report_fixture = make_generate_html_report_fixture(_COMPARISON_SCORES, Path(__file__).stem + ".html")
_attach_comparison_properties_fixture = make_attach_comparison_properties_fixture(_COMPARISON_SCORES)


@pytest.mark.parametrize("scene_name,physics_backend,renderer", GAUSSIAN_SPLATS_KIT_RENDERING_PARAMS)
def test_rendering_gaussian_splats(scene_name, physics_backend, renderer):
    """Gaussian scene RGB output must match golden images (Isaac RTX and Newton Warp)."""
    rendering_test_gaussian_splats(scene_name, physics_backend, renderer, _COMPARISON_SCORES)
