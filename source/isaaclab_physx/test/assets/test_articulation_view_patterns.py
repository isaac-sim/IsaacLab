# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for multi-wildcard articulation view pattern resolution (no simulation)."""

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).parents[2] / "isaaclab_physx" / "assets" / "articulation" / "view_patterns.py"
_spec = importlib.util.spec_from_file_location("view_patterns", _MODULE_PATH)
view_patterns = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(view_patterns)


def test_single_wildcard_passes_through():
    result = view_patterns.resolve_view_path_patterns("/World/envs/env_.*/Robot", lambda expr: [])
    assert result == "/World/envs/env_*/Robot"


def test_multi_wildcard_expands_to_per_subasset_patterns():
    matches = [
        "/World/envs/env_0/Rig/parts/part_a",
        "/World/envs/env_0/Rig/parts/part_b",
        "/World/envs/env_1/Rig/parts/part_a",
        "/World/envs/env_1/Rig/parts/part_b",
    ]
    result = view_patterns.resolve_view_path_patterns("/World/envs/env_.*/Rig/parts/part_.*", lambda expr: matches)
    assert result == [
        "/World/envs/env_*/Rig/parts/part_a",
        "/World/envs/env_*/Rig/parts/part_b",
    ]


def test_multi_wildcard_in_middle_segment():
    matches = [
        "/World/envs/env_0/rig_x/base",
        "/World/envs/env_1/rig_x/base",
        "/World/envs/env_0/rig_y/base",
        "/World/envs/env_1/rig_y/base",
    ]
    result = view_patterns.resolve_view_path_patterns("/World/envs/env_.*/rig_.*/base", lambda expr: matches)
    assert result == [
        "/World/envs/env_*/rig_x/base",
        "/World/envs/env_*/rig_y/base",
    ]


def test_heterogeneous_subassets_raise():
    matches = [
        "/World/envs/env_0/Rig/parts/part_a",
        "/World/envs/env_1/Rig/parts/part_b",
    ]
    with pytest.raises(RuntimeError, match="different sub-asset"):
        view_patterns.resolve_view_path_patterns("/World/envs/env_.*/Rig/parts/part_.*", lambda expr: matches)


def test_no_matches_raise():
    with pytest.raises(RuntimeError, match="No prims match"):
        view_patterns.resolve_view_path_patterns("/World/envs/env_.*/Rig/parts/part_.*", lambda expr: [])
