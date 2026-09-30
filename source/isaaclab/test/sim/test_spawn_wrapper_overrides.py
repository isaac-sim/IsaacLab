# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.sim.spawners.wrappers import MultiAssetSpawnerCfg, wrappers


@pytest.mark.parametrize(
    "global_value,child_value,expected",
    [(None, True, True), (True, False, True), (False, True, False)],
)
def test_multi_asset_contact_reporting_override(monkeypatch, global_value, child_value, expected):
    """An omitted global setting preserves each asset's contact reporting choice."""
    observed = []

    def record_spawn(prim_path, cfg, **kwargs):
        observed.append(cfg.activate_contact_sensors)

    asset_cfg = UsdFileCfg(usd_path="unused.usd", activate_contact_sensors=child_value)
    asset_cfg.func = record_spawn
    global_kwargs = {} if global_value is None else {"activate_contact_sensors": global_value}
    cfg = MultiAssetSpawnerCfg(
        assets_cfg=[asset_cfg], spawn_paths=["/World/Asset"], random_choice=False, **global_kwargs
    )
    monkeypatch.setattr(wrappers.sim_utils, "find_first_matching_prim", lambda path: path)

    wrappers.spawn_multi_asset("/World/Asset", cfg)

    assert observed == [expected]
