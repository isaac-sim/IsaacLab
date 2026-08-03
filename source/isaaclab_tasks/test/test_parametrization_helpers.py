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
    make_kitless_rendering_params,
    make_kitless_rendering_params_franka,
    make_kitless_rendering_params_lift,
    make_skip_rendering_params,
    make_xfail_rendering_params,
)


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


def test_lift_factory_applies_shared_native_crash_policy() -> None:
    """Both stage variants should share the ticketed Lift MDL skips."""
    params = {param.id: param for param in make_kitless_rendering_params_lift()}

    for variant in ("legacy", "ovstage"):
        for data_type in ("simple_shading_diffuse_mdl", "simple_shading_full_mdl"):
            param = params[f"{variant}-newton-ovrtx-{data_type}"]
            assert [mark.name for mark in param.marks] == ["skip"]
            assert "NVBUG#6524987" in param.marks[0].kwargs["reason"]


def test_franka_factory_adds_cloth_only_motion_policy() -> None:
    """Franka suites should share table xfails while cloth adds motion-vector xfails."""
    soft_params = {param.id: param for param in make_kitless_rendering_params_franka()}
    cloth_params = {
        param.id: param for param in make_kitless_rendering_params_franka(include_cloth_motion_vectors=True)
    }

    table_id = "legacy-newton-newton_warp-rgb"
    assert [mark.name for mark in soft_params[table_id].marks] == ["xfail"]
    assert [mark.name for mark in cloth_params[table_id].marks] == ["xfail"]
    assert "OMPE-103086" in soft_params[table_id].marks[0].kwargs["reason"]

    for variant in ("legacy", "ovstage"):
        motion_id = f"{variant}-newton-ovrtx-motion_vectors"
        assert "xfail" not in [mark.name for mark in soft_params[motion_id].marks]
        assert [mark.name for mark in cloth_params[motion_id].marks] == ["xfail"]
        assert "NVBUG#6489754" in cloth_params[motion_id].marks[0].kwargs["reason"]


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
