# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less rendering correctness tests for Dexsuite KukaAllegro Lift backend combinations."""

from pathlib import Path

import pytest
from rendering_test_utils import (
    KITLESS_PHYSICS_RENDERER_AOV_COMBINATIONS,
    make_attach_comparison_properties_fixture,
    make_determinism_fixture,
    make_generate_html_report_fixture,
    make_kitless_rendering_params,
    make_require_ovlibs_install_fixture,
    make_skip_rendering_params,
    rendering_test_dexsuite_kuka,
)

pytestmark = [pytest.mark.isaacsim_ci, pytest.mark.arm_ci]

_DEXSUITE_RENDERER_CRASH_SKIP_REASON = "Dexsuite kitless OVRTX rendering may crash or time out (NVBUG#6524987)."
_RENDERING_PARAMS = make_skip_rendering_params(
    make_kitless_rendering_params(KITLESS_PHYSICS_RENDERER_AOV_COMBINATIONS),
    {
        (variant, "newton", "ovrtx_renderer", data_type): _DEXSUITE_RENDERER_CRASH_SKIP_REASON
        for variant in ("legacy", "ovstage")
        for data_type in ("simple_shading_diffuse_mdl", "simple_shading_full_mdl")
    },
)
_COMPARISON_SCORES: list[dict] = []

_determinism_fixture = make_determinism_fixture()
_generate_html_report_fixture = make_generate_html_report_fixture(_COMPARISON_SCORES, Path(__file__).stem + ".html")
_attach_comparison_properties_fixture = make_attach_comparison_properties_fixture(_COMPARISON_SCORES)
_require_ovlibs_install_fixture = make_require_ovlibs_install_fixture()


@pytest.mark.parametrize(
    "ovstage_variant,physics_backend,renderer,data_type", _RENDERING_PARAMS, indirect=["ovstage_variant"]
)
def test_rendering_dexsuite_kuka_homo_kitless(ovstage_variant, physics_backend, renderer, data_type):
    """Camera output must match golden images (Dexsuite KukaAllegro Lift, single camera)."""
    rendering_test_dexsuite_kuka(physics_backend, renderer, data_type, True, _COMPARISON_SCORES)
