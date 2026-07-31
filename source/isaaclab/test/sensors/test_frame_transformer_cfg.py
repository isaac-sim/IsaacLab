# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

"""Everything else follows."""

import pytest

from isaaclab.sensors.frame_transformer.base_frame_transformer import BaseFrameTransformer


@pytest.mark.parametrize(
    ("prim_path", "prim_path_regex", "expected"),
    [
        ("/Robot", None, "/Robot"),
        (None, "/Robot/pelvis", "/Robot/pelvis"),
    ],
)
def test_select_prim_path(prim_path, prim_path_regex, expected):
    assert BaseFrameTransformer._select_prim_path(prim_path, prim_path_regex) == expected


@pytest.mark.parametrize(
    ("prim_path", "prim_path_regex"),
    [
        (None, None),
        ("/Robot", "/Robot"),
    ],
)
def test_select_prim_path_rejects_both_or_neither(prim_path, prim_path_regex):
    with pytest.raises(ValueError, match="exactly one"):
        BaseFrameTransformer._select_prim_path(prim_path, prim_path_regex)
