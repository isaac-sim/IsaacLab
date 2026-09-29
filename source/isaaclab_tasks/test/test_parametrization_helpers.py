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
    attach_comparison_properties,
    generate_html_report,
    group_rendering_params,
    make_kitless_rendering_params,
    make_skip_rendering_params,
)


def test_group_rendering_params_groups_static_data_types_with_matching_marks() -> None:
    """Static AOVs with the same rendering configuration and marks should share a case, once per renderer."""
    flaky = pytest.mark.flaky(max_runs=3, min_passes=1)
    params = [
        pytest.param("physx", "isaacsim_rtx_renderer", "albedo", id="physx-rtx-albedo", marks=flaky),
        pytest.param("physx", "isaacsim_rtx_renderer", "normals", id="physx-rtx-normals", marks=flaky),
        pytest.param(
            "physx",
            "isaacsim_rtx_renderer",
            "instance_segmentation",
            id="physx-rtx-instance",
            marks=pytest.mark.xfail(reason="Known segmentation regression."),
        ),
        pytest.param("physx", "newton_renderer", "rgb", id="physx-warp-rgb"),
        pytest.param("physx", "newton_renderer", "depth", id="physx-warp-depth"),
    ]

    grouped = group_rendering_params(params)

    assert [tuple(param.values) for param in grouped] == [
        ("physx", "isaacsim_rtx_renderer", ["albedo", "normals"]),
        ("physx", "isaacsim_rtx_renderer", ["instance_segmentation"]),
        ("physx", "newton_renderer", ["rgb", "depth"]),
    ]
    assert [param.id for param in grouped] == [
        "physx-isaacsim_rtx_renderer-static",
        "physx-rtx-instance",
        "physx-newton_renderer-static",
    ]
    assert [[mark.name for mark in param.marks] for param in grouped] == [["flaky"], ["xfail"], []]


def test_group_rendering_params_isolates_temporal_and_minimal_data_types() -> None:
    """AOVs requiring distinct capture or render-product state should stay isolated."""
    flaky = pytest.mark.flaky(max_runs=3, min_passes=1)
    params = [
        pytest.param("physx", "isaacsim_rtx_renderer", "rgb", id="physx-rtx-rgb", marks=flaky),
        pytest.param("physx", "isaacsim_rtx_renderer", "depth", id="physx-rtx-depth", marks=flaky),
        pytest.param(
            "physx",
            "isaacsim_rtx_renderer",
            "distance_to_image_plane",
            id="physx-rtx-distance_to_image_plane",
            marks=flaky,
        ),
        pytest.param("physx", "isaacsim_rtx_renderer", "motion_vectors", id="physx-rtx-motion", marks=flaky),
        pytest.param(
            "physx", "isaacsim_rtx_renderer", "simple_shading_diffuse_mdl", id="physx-rtx-diffuse_mdl", marks=flaky
        ),
        pytest.param("physx", "isaacsim_rtx_renderer", "simple_shading_full_mdl", id="physx-rtx-full_mdl", marks=flaky),
    ]

    grouped = group_rendering_params(params)

    assert [tuple(param.values) for param in grouped] == [
        ("physx", "isaacsim_rtx_renderer", ["rgb", "depth", "distance_to_image_plane"]),
        ("physx", "isaacsim_rtx_renderer", ["motion_vectors"]),
        ("physx", "isaacsim_rtx_renderer", ["simple_shading_diffuse_mdl"]),
        ("physx", "isaacsim_rtx_renderer", ["simple_shading_full_mdl"]),
    ]
    assert [param.id for param in grouped] == [
        "physx-isaacsim_rtx_renderer-static",
        "physx-rtx-motion",
        "physx-rtx-diffuse_mdl",
        "physx-rtx-full_mdl",
    ]


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
