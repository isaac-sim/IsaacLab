# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less rendering correctness tests for test-local Franka cloth camera setup."""

from pathlib import Path

import pytest
from rendering_test_utils import (
    KITLESS_PHYSICS_RENDERER_AOV_COMBINATIONS,
    make_attach_comparison_properties_fixture,
    make_determinism_fixture,
    make_generate_html_report_fixture,
    make_require_ovlibs_install_fixture,
    make_xfail_rendering_params,
    rendering_test_franka_cloth,
)

pytestmark = pytest.mark.isaacsim_ci

_RENDERING_PARAMS = make_xfail_rendering_params(
    KITLESS_PHYSICS_RENDERER_AOV_COMBINATIONS,
    {("newton", "ovrtx_renderer", "motion_vectors"): ("OVRTX 0.4 omits deformable motion vectors (NVBUG#6489754).")},
)
_COMPARISON_SCORES: list[dict] = []

_determinism_fixture = make_determinism_fixture()
_generate_html_report_fixture = make_generate_html_report_fixture(_COMPARISON_SCORES, Path(__file__).stem + ".html")
_attach_comparison_properties_fixture = make_attach_comparison_properties_fixture(_COMPARISON_SCORES)
_require_ovlibs_install_fixture = make_require_ovlibs_install_fixture()


@pytest.mark.parametrize("physics_backend,renderer,data_type", _RENDERING_PARAMS)
def test_rendering_franka_cloth_kitless(ovstage_variant, physics_backend, renderer, data_type):
    """Camera output must match golden images for the Franka cloth test setup."""
    rendering_test_franka_cloth(physics_backend, renderer, data_type, _COMPARISON_SCORES)
