# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less rendering correctness tests for Cartpole environment backend combinations."""

from pathlib import Path

import pytest
from rendering_test_utils import (
    KITLESS_PHYSICS_RENDERER_AOV_COMBINATIONS,
    _make_sensor_data_type_params,
    group_rendering_params,
    make_attach_comparison_properties_fixture,
    make_determinism_fixture,
    make_generate_html_report_fixture,
    make_kitless_rendering_params,
    make_require_ovlibs_install_fixture,
    rendering_test_cartpole,
)

# Async variants of the synchronous combinations, so they must match the same golden images: the
# frame of latency is absorbed by the tolerances, and the harness's warm-up frames prime the pipeline.
# Restricted to the legacy stage path: the ovstage path renders synchronously even when async is
# requested, so an ovstage lane here would only duplicate the synchronous ovstage coverage.
_ASYNC_COMBINATIONS = [
    param
    for param in group_rendering_params(
        make_kitless_rendering_params(
            [
                *_make_sensor_data_type_params("ovphysx", "ovrtx", ["rgb"]),
                *_make_sensor_data_type_params("newton", "ovrtx", ["rgb"]),
            ]
        )
    )
    if param.values[0] == "legacy"
]

pytestmark = pytest.mark.arm_ci

_RENDERING_PARAMS = group_rendering_params(make_kitless_rendering_params(KITLESS_PHYSICS_RENDERER_AOV_COMBINATIONS))
_COMPARISON_SCORES: list[dict] = []

_determinism_fixture = make_determinism_fixture()
_generate_html_report_fixture = make_generate_html_report_fixture(_COMPARISON_SCORES, Path(__file__).stem + ".html")
_attach_comparison_properties_fixture = make_attach_comparison_properties_fixture(_COMPARISON_SCORES)
_require_ovlibs_install_fixture = make_require_ovlibs_install_fixture()


@pytest.mark.parametrize(
    "ovstage_variant,physics_backend,renderer,data_types", _RENDERING_PARAMS, indirect=["ovstage_variant"]
)
def test_rendering_cartpole_kitless(ovstage_variant, physics_backend, renderer, data_types):
    """Camera output must match golden images (Cartpole camera presets env)."""
    rendering_test_cartpole(physics_backend, renderer, data_types, _COMPARISON_SCORES)


@pytest.mark.parametrize(
    "ovstage_variant,physics_backend,renderer,data_types", _ASYNC_COMBINATIONS, indirect=["ovstage_variant"]
)
def test_rendering_cartpole_kitless_async(ovstage_variant, physics_backend, renderer, data_types, monkeypatch):
    """OVRTX async-rendered camera output must match the synchronous golden images (within tolerance)."""
    monkeypatch.setenv("ISAAC_LAB_ASYNC_RENDERING", "1")
    rendering_test_cartpole(physics_backend, renderer, data_types, _COMPARISON_SCORES)
