# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for FrameTransformer._validate_site_map."""

import pytest
from isaaclab_newton.sensors.frame_transformer.frame_transformer import FrameTransformer


def _make_site_map(
    source_per_world: list[list[int]],
    target_per_worlds: list[list[list[int]]],
    world_origin_idx: int = 0,
) -> dict:
    m = {
        "world_origin": (world_origin_idx, None),
        "source": (None, source_per_world),
    }
    for i, pw in enumerate(target_per_worlds):
        m[f"target_{i}"] = (None, pw)
    return m


class TestSourceValidation:
    def test_valid_source_one_per_env(self):
        site_map = _make_site_map([[10], [20]], [])
        indices, _ = FrameTransformer._validate_site_map("source", "/Robot/base", [], [], site_map, num_envs=2)
        assert indices == [10, 20]

    def test_source_wrong_env_count_raises(self):
        # site map has 1 world entry but num_envs=2
        site_map = _make_site_map([[10]], [])
        with pytest.raises(ValueError, match="1 world entries.*expected 2"):
            FrameTransformer._validate_site_map("source", "/Robot/base", [], [], site_map, num_envs=2)

    def test_source_zero_in_env_raises(self):
        site_map = _make_site_map([[], [20]], [])
        with pytest.raises(ValueError, match="matched 0 bodies in env 0"):
            FrameTransformer._validate_site_map("source", "/Robot/base", [], [], site_map, num_envs=2)

    def test_source_two_in_env_raises(self):
        site_map = _make_site_map([[10, 11], [20]], [])
        with pytest.raises(ValueError, match="matched 2 bodies in env 0"):
            FrameTransformer._validate_site_map("source", "/Robot/base", [], [], site_map, num_envs=2)


class TestTargetValidation:
    def test_valid_single_target_per_env(self):
        site_map = _make_site_map([[10], [20]], [[[30], [40]]])
        _, tgt = FrameTransformer._validate_site_map(
            "source", "/Robot/base", ["target_0"], ["/Robot/hand"], site_map, num_envs=2
        )
        assert tgt[0] == [[30], [40]]

    def test_valid_wildcard_two_bodies_per_env(self):
        site_map = _make_site_map([[10], [20]], [[[30, 31], [40, 41]]])
        _, tgt = FrameTransformer._validate_site_map(
            "source", "/Robot/base", ["target_0"], ["/Robot/foot.*"], site_map, num_envs=2
        )
        assert tgt[0] == [[30, 31], [40, 41]]

    def test_target_zero_bodies_raises(self):
        site_map = _make_site_map([[10], [20]], [[[], []]])
        with pytest.raises(ValueError, match="matched no bodies"):
            FrameTransformer._validate_site_map(
                "source", "/Robot/base", ["target_0"], ["/Robot/foot.*"], site_map, num_envs=2
            )

    def test_target_non_uniform_raises(self):
        site_map = _make_site_map([[10], [20]], [[[30, 31], [40]]])
        with pytest.raises(ValueError, match="different numbers of bodies"):
            FrameTransformer._validate_site_map(
                "source", "/Robot/base", ["target_0"], ["/Robot/foot.*"], site_map, num_envs=2
            )
