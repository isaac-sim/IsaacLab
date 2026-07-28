# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for rendering correctness parameter helpers."""

import pytest
from rendering_test_utils import (
    make_kitless_rendering_params,
    make_skip_rendering_params,
    make_xfail_rendering_params,
)


def test_make_kitless_rendering_params_expands_only_ovrtx() -> None:
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


def test_make_xfail_rendering_params_removes_flaky_mark() -> None:
    params = [
        pytest.param(
            "ovstage",
            "newton",
            "ovrtx_renderer",
            "albedo",
            id="ovstage-newton-ovrtx-albedo",
            marks=pytest.mark.flaky(max_runs=3, min_passes=1),
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
