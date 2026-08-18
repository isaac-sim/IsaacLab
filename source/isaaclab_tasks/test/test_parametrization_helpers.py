# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for rendering correctness parameter helpers."""

from pathlib import Path
from unittest.mock import Mock

import pytest
import rendering_test_utils
from rendering_test_utils import (
    KITLESS_PHYSICS_RENDERER_AOV_COMBINATIONS,
    attach_comparison_properties,
    generate_html_report,
    make_kitless_rendering_params,
    make_kitless_rendering_params_franka,
    make_kitless_rendering_params_lift,
    make_skip_rendering_params,
    make_xfail_rendering_params,
)


@pytest.mark.parametrize(
    ("env_name", "renderer", "data_type", "expected"),
    [
        ("franka_soft", "ovrtx_renderer", "albedo", 3.0),
        ("franka_soft", "isaacsim_rtx_renderer", "albedo", 8.0),
        ("cartpole", "ovrtx_renderer", "rgb", 1.5),
        ("shadow_hand", "ovrtx_renderer", "depth", 5.0),
    ],
)
def test_ovrtx_image_difference_threshold_is_capped(
    env_name: str, renderer: str, data_type: str, expected: float
) -> None:
    """OVRTX should use a tighter cap without loosening stricter environment thresholds."""
    assert rendering_test_utils._max_different_pixels_percentage(env_name, renderer, data_type) == expected


@pytest.mark.parametrize(
    ("renderer", "expected_steps"),
    [
        ("ovrtx_renderer", 3),
        ("isaacsim_rtx_renderer", 2),
    ],
)
def test_motion_history_steps(renderer: str, expected_steps: int) -> None:
    """OVRTX should receive one extra motion-history step."""
    env = Mock()
    env.action_space.shape = (1,)
    env.device = "cpu"

    rendering_test_utils.maybe_step_env_for_motion(env, renderer, "motion_vectors")

    assert env.step.call_count == expected_steps


def test_make_kitless_rendering_params_expands_only_ovrtx() -> None:
    """OVStage variants should be emitted only for the OVRTX renderer."""
    params = [
        pytest.param("newton", "ovrtx_renderer", "rgb", id="newton-ovrtx-rgb"),
        pytest.param("newton", "newton_renderer", "rgb", id="newton-newton_warp-rgb"),
    ]

    expanded = make_kitless_rendering_params(params)

    assert [param.id for param in expanded] == [
        "legacy-newton-ovrtx-rgb",
        "ovstage-newton-ovrtx-rgb",
        "legacy-newton-newton_warp-rgb",
    ]
    assert [tuple(param.values) for param in expanded] == [
        ("legacy", "newton", "ovrtx_renderer", "rgb"),
        ("ovstage", "newton", "ovrtx_renderer", "rgb"),
        ("legacy", "newton", "newton_renderer", "rgb"),
    ]


def test_make_xfail_rendering_params_replaces_flaky_and_xfail_marks() -> None:
    """Expected failures should run once with one current reason."""
    params = [
        pytest.param(
            "ovstage",
            "newton",
            "ovrtx_renderer",
            "albedo",
            id="ovstage-newton-ovrtx-albedo",
            marks=[
                pytest.mark.flaky(max_runs=3, min_passes=1),
                pytest.mark.xfail(reason="Obsolete rendering regression.", strict=False),
            ],
        )
    ]

    marked = make_xfail_rendering_params(
        params,
        {("ovstage", "newton", "ovrtx_renderer", "albedo"): "Known rendering regression."},
    )

    assert [mark.name for mark in marked[0].marks] == ["xfail"]
    xfail_mark = marked[0].marks[0]
    assert xfail_mark.kwargs["reason"] == "Known rendering regression."
    assert xfail_mark.kwargs["strict"] is False


def test_make_skip_rendering_params_overrides_xfail_and_flaky_marks() -> None:
    """Native-crash skips should override inherited retry and xfail marks."""
    params = [
        pytest.param(
            "legacy",
            "newton",
            "ovrtx_renderer",
            "simple_shading_full_mdl",
            id="legacy-newton-ovrtx-simple_shading_full_mdl",
            marks=[
                pytest.mark.flaky(max_runs=3, min_passes=1),
                pytest.mark.xfail(reason="Known image mismatch.", strict=False),
            ],
        )
    ]

    marked = make_skip_rendering_params(
        params,
        {("legacy", "newton", "ovrtx_renderer", "simple_shading_full_mdl"): "Native renderer crash."},
    )

    assert [mark.name for mark in marked[0].marks] == ["skip"]
    assert marked[0].marks[0].kwargs["reason"] == "Native renderer crash."


def test_kitless_matrix_has_no_ovrtx_041_xfails() -> None:
    """OVRTX 0.4.1 textured and motion AOVs should run without release xfails."""
    params = {param.id: param for param in KITLESS_PHYSICS_RENDERER_AOV_COMBINATIONS}

    for data_type in ("albedo", "simple_shading_diffuse_mdl", "simple_shading_full_mdl"):
        for physics_backend in ("newton", "ovphysx"):
            param = params[f"{physics_backend}-ovrtx-{data_type}"]
            assert "xfail" not in [mark.name for mark in param.marks]

    expanded = {param.id: param for param in make_kitless_rendering_params(list(params.values()))}
    assert "xfail" not in [mark.name for mark in expanded["ovstage-ovphysx-ovrtx-motion_vectors"].marks]


def test_lift_factory_retains_retries_without_native_crash_skips() -> None:
    """Lift OVRTX MDL cases should run with the shared retry policy."""
    params = {param.id: param for param in make_kitless_rendering_params_lift()}

    for variant in ("legacy", "ovstage"):
        for physics_backend in ("newton", "ovphysx"):
            for data_type in ("simple_shading_diffuse_mdl", "simple_shading_full_mdl"):
                param = params[f"{variant}-{physics_backend}-ovrtx-{data_type}"]
                assert [mark.name for mark in param.marks] == ["flaky"]

    # Lift OVPhysX albedo passes, so it must not inherit an unrelated exemption.
    assert "xfail" not in [mark.name for mark in params["legacy-ovphysx-ovrtx-albedo"].marks]


def test_franka_factory_has_no_cloth_motion_xfail() -> None:
    """OVRTX 0.4.1 cloth motion vectors should run without an xfail."""
    params = {param.id: param for param in make_kitless_rendering_params_franka()}

    for variant in ("legacy", "ovstage"):
        motion_id = f"{variant}-newton-ovrtx-motion_vectors"
        assert "xfail" not in [mark.name for mark in params[motion_id].marks]


def test_html_report_labels_xfail_and_xpass_outcomes(monkeypatch, tmp_path: Path) -> None:
    """Expected failures and unexpected passes should be distinct in HTML."""
    reason = "Known <b>rendering</b> regression (NVBUG#1234567)."
    comparison_scores = [
        {
            "test": "cartpole",
            "backend": "newton",
            "renderer": "ovrtx_renderer",
            "ovstage_variant": "Yes",
            "aov": "albedo",
            "diff_pct": 12.5,
            "threshold": 1.5,
            "ssim": 0.9,
            "ssim_threshold": 0.985,
            "ssim_checked": True,
            "passed": False,
        },
        {
            "test": "cartpole",
            "backend": "newton",
            "renderer": "ovrtx_renderer",
            "ovstage_variant": "Yes",
            "aov": "rgb",
            "diff_pct": 0.0,
            "threshold": 1.5,
            "ssim": 1.0,
            "ssim_threshold": 0.985,
            "ssim_checked": True,
            "passed": True,
        },
    ]
    node = Mock()
    node.user_properties = []
    node.get_closest_marker.return_value = pytest.mark.xfail(reason=reason, strict=False).mark
    request = Mock(node=node)

    attach_comparison_properties(request, comparison_scores, initial_count=0)
    comparison_scores.append(
        {
            "test": "shadow_hand",
            "backend": "newton",
            "renderer": "ovrtx_renderer",
            "ovstage_variant": "Yes",
            "aov": "albedo",
            "diff_pct": 0.0,
            "threshold": 5.0,
            "ssim": 1.0,
            "ssim_threshold": 0.985,
            "ssim_checked": True,
            "passed": True,
        }
    )
    attach_comparison_properties(request, comparison_scores, initial_count=2)
    comparison_scores.append(
        {
            "test": "ordinary",
            "backend": "newton",
            "renderer": "ovrtx_renderer",
            "ovstage_variant": "Yes",
            "aov": "depth",
            "diff_pct": 50.0,
            "threshold": 5.0,
            "ssim": 0.5,
            "ssim_threshold": 0.985,
            "ssim_checked": False,
            "passed": False,
        }
    )
    monkeypatch.setattr(rendering_test_utils, "_COMPARISON_IMAGES_DIR", str(tmp_path))
    generate_html_report(comparison_scores, "report.html")

    report = (tmp_path / "report.html").read_text(encoding="utf-8")
    escaped_reason = "Known &lt;b&gt;rendering&lt;/b&gt; regression (NVBUG#1234567)."
    assert report.count('<td class="status-unreliable">UNRELIABLE (XFAIL)</td>') == 2
    assert report.count('<td class="status-xpass">XPASS (REVIEW XFAIL)</td>') == 1
    assert report.count(escaped_reason) == 3
    assert reason not in report
    assert report.index(escaped_reason) < report.index("<td>ordinary</td>")
