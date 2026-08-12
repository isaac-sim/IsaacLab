# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less rendering correctness tests for Cartpole environment backend combinations."""

import tempfile
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
    rendering_test_cartpole,
)

pytestmark = [pytest.mark.isaacsim_ci, pytest.mark.arm_ci]

_OVRTX_TEXTURE_READINESS_XFAIL_REASON = "OVRTX 0.4 may return before textured materials are ready (NVBUG#6505191)."
_OVSTAGE_OVPHYSX_MOTION_XFAIL_REASON = (
    "OVStage-backed OVRTX motion vectors do not yet match the legacy OVRTX path with OVPhysX."
)
_BASE_RENDERING_PARAMS = make_kitless_rendering_params(
    make_xfail_rendering_params(
        KITLESS_PHYSICS_RENDERER_AOV_COMBINATIONS,
        {
            ("ovphysx", "ovrtx_renderer", data_type): _OVRTX_TEXTURE_READINESS_XFAIL_REASON
            for data_type in ("albedo", "simple_shading_diffuse_mdl", "simple_shading_full_mdl")
        },
    )
)
_RENDERING_PARAMS = make_xfail_rendering_params(
    _BASE_RENDERING_PARAMS,
    {("ovstage", "ovphysx", "ovrtx_renderer", "motion_vectors"): _OVSTAGE_OVPHYSX_MOTION_XFAIL_REASON},
)
_COMPARISON_SCORES: list[dict] = []

_determinism_fixture = make_determinism_fixture()
_generate_html_report_fixture = make_generate_html_report_fixture(_COMPARISON_SCORES, Path(__file__).stem + ".html")
_attach_comparison_properties_fixture = make_attach_comparison_properties_fixture(_COMPARISON_SCORES)
_require_ovlibs_install_fixture = make_require_ovlibs_install_fixture()


@pytest.mark.parametrize(
    "ovstage_variant,physics_backend,renderer,data_type", _RENDERING_PARAMS, indirect=["ovstage_variant"]
)
def test_rendering_cartpole_kitless(ovstage_variant, physics_backend, renderer, data_type):
    """Camera output must match golden images (Cartpole camera presets env)."""
    rendering_test_cartpole(physics_backend, renderer, data_type, _COMPARISON_SCORES)


# TEMPORARY CI PROBE - DO NOT MERGE. Every job on #7035 was green, and the renderer-log replay added
# there only surfaces in a failing test's report, so nothing in CI has yet shown a real OVRTX log
# being routed. This forces one failure after a real OVRTX render. Runs last so the replay has a
# non-zero start offset and a log large enough to hit the 64 KiB cap - the interesting cases.
# Deliberately carries no flaky mark: a retried failure would report through the flaky plugin instead
# of the normal report where the captured output lives.
@pytest.mark.parametrize(
    "ovstage_variant,physics_backend,renderer,data_type",
    [pytest.param("legacy", "ovphysx", "ovrtx_renderer", "rgb", id="legacy-ovrtx-log-probe")],
    indirect=["ovstage_variant"],
)
def test_ovrtx_log_ci_probe(ovstage_variant, physics_backend, renderer, data_type):
    """Fail on purpose so the job log shows the replayed OVRTX renderer log."""
    rendering_test_cartpole(physics_backend, renderer, data_type, _COMPARISON_SCORES)

    # Reported from inside the test process so an empty replay section can be told apart from a log
    # the renderer never wrote - a path that is missing or unwritable under the container's uid.
    log_path = Path(tempfile.gettempdir()) / "ovrtx_renderer.log"
    size = log_path.stat().st_size if log_path.exists() else -1
    pytest.fail(f"CI probe: log_path={log_path} exists={log_path.exists()} size={size}")
