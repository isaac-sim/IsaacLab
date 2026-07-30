# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for rendering correctness parameter helpers."""

import pytest
from rendering_test_utils import (
    make_kitless_rendering_params,
    make_kitless_rendering_params_dexsuite,
    make_kitless_rendering_params_franka,
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


def test_dexsuite_factory_applies_shared_native_crash_policy() -> None:
    """Both stage variants should share the ticketed Dexsuite MDL skips."""
    params = {param.id: param for param in make_kitless_rendering_params_dexsuite()}

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
