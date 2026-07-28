# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less rendering correctness tests for Cartpole environment backend combinations."""

from pathlib import Path

import pytest
from rendering_test_utils import (
    KITLESS_PHYSICS_RENDERER_AOV_COMBINATIONS,
    make_attach_comparison_properties_fixture,
    make_determinism_fixture,
    make_generate_html_report_fixture,
    make_require_ovlibs_install_fixture,
    make_xfail_rendering_params,
    rendering_test_cartpole,
)

pytestmark = [pytest.mark.isaacsim_ci, pytest.mark.arm_ci]

_OVRTX_TEXTURE_READINESS_XFAIL_REASON = "OVRTX 0.4 may return before textured materials are ready (NVBUG#6505191)."
_OVSTAGE_OVPHYSX_MOTION_XFAIL_REASON = (
    "OVStage-backed OVRTX motion vectors do not yet match the legacy OVRTX path with OVPhysX."
)
_RENDERING_PARAMS = make_xfail_rendering_params(
    KITLESS_PHYSICS_RENDERER_AOV_COMBINATIONS,
    {
        ("ovphysx", "ovrtx_renderer", data_type): _OVRTX_TEXTURE_READINESS_XFAIL_REASON
        for data_type in ("albedo", "simple_shading_diffuse_mdl", "simple_shading_full_mdl")
    },
)
_COMPARISON_SCORES: list[dict] = []

_determinism_fixture = make_determinism_fixture()
_generate_html_report_fixture = make_generate_html_report_fixture(_COMPARISON_SCORES, Path(__file__).stem + ".html")
_attach_comparison_properties_fixture = make_attach_comparison_properties_fixture(_COMPARISON_SCORES)
_require_ovlibs_install_fixture = make_require_ovlibs_install_fixture()


@pytest.mark.parametrize("physics_backend,renderer,data_type", _RENDERING_PARAMS)
def test_rendering_cartpole_kitless(ovstage_variant, physics_backend, renderer, data_type):
    """Camera output must match golden images (Cartpole camera presets env)."""
    if (
        ovstage_variant == "ovstage"
        and physics_backend == "ovphysx"
        and renderer == "ovrtx_renderer"
        and data_type == "motion_vectors"
    ):
        pytest.xfail(_OVSTAGE_OVPHYSX_MOTION_XFAIL_REASON)
    rendering_test_cartpole(physics_backend, renderer, data_type, _COMPARISON_SCORES)
