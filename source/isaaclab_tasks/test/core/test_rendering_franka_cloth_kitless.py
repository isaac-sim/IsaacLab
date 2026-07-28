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
    make_kitless_rendering_params,
    make_require_ovlibs_install_fixture,
    make_xfail_rendering_params,
    rendering_test_franka_cloth,
)

pytestmark = pytest.mark.isaacsim_ci

_NEWTON_WARP_MISSING_TABLE_XFAIL_REASON = "Missing table in Newton Warp renderer (OMPE-103086)."
_OVRTX_CLOTH_MOTION_XFAIL_REASON = "Missing cloth in OVRTX 0.4 motion vectors (NVBUG#6489754)."
_BASE_RENDERING_PARAMS = make_kitless_rendering_params(KITLESS_PHYSICS_RENDERER_AOV_COMBINATIONS)
_EXPECTED_FAILURES = {
    tuple(param.values): _NEWTON_WARP_MISSING_TABLE_XFAIL_REASON
    for param in _BASE_RENDERING_PARAMS
    if param.values[1] == "newton" and param.values[2] == "newton_renderer"
}
_EXPECTED_FAILURES.update(
    {
        (variant, "newton", "ovrtx_renderer", "motion_vectors"): _OVRTX_CLOTH_MOTION_XFAIL_REASON
        for variant in ("legacy", "ovstage")
    }
)
_RENDERING_PARAMS = make_xfail_rendering_params(_BASE_RENDERING_PARAMS, _EXPECTED_FAILURES)
_COMPARISON_SCORES: list[dict] = []

_determinism_fixture = make_determinism_fixture()
_generate_html_report_fixture = make_generate_html_report_fixture(_COMPARISON_SCORES, Path(__file__).stem + ".html")
_attach_comparison_properties_fixture = make_attach_comparison_properties_fixture(_COMPARISON_SCORES)
_require_ovlibs_install_fixture = make_require_ovlibs_install_fixture()


@pytest.mark.parametrize(
    "ovstage_variant,physics_backend,renderer,data_type", _RENDERING_PARAMS, indirect=["ovstage_variant"]
)
def test_rendering_franka_cloth_kitless(ovstage_variant, physics_backend, renderer, data_type):
    """Camera output must match golden images for the Franka cloth test setup."""
    rendering_test_franka_cloth(physics_backend, renderer, data_type, _COMPARISON_SCORES)
