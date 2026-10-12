# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Newton asset source paths."""

import pytest

from isaaclab.utils.assets import NEWTON_ASSET_DIR

from isaaclab_assets.robots.dr_legs import DR_LEGS_IMPLICIT_PD_CFG
from isaaclab_assets.robots.fourbar_pole import FOURBAR_POLE_CFG


@pytest.mark.parametrize(
    ("asset_cfg", "relative_path"),
    [
        (DR_LEGS_IMPLICIT_PD_CFG, "disneyresearch/dr_legs/usd/dr_legs.usda"),
        (FOURBAR_POLE_CFG, "fourbar_pole/usd/fourbar_pole.usda"),
    ],
)
def test_newton_asset_loading_is_deferred(asset_cfg, relative_path: str):
    """Keep the source URL in the config so loading happens when the asset spawns."""
    assert asset_cfg.spawn.usd_path == f"{NEWTON_ASSET_DIR}/{relative_path}"
