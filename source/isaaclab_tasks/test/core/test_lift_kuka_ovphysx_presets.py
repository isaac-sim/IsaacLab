# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Lift Kuka Allegro OVPhysX object preset compatibility."""

import sys

import pytest

from isaaclab.sim import MultiAssetSpawnerCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config


@pytest.mark.parametrize(
    "task_name,preset_args",
    [
        ("Isaac-Lift-KukaAllegro", ("physics=ovphysx", "presets=shapes")),
        ("Isaac-Reorient-KukaAllegro", ("presets=shapes", "physics=ovphysx")),
    ],
)
def test_ovphysx_accepts_heterogeneous_shapes_independent_of_argument_order(
    task_name: str, preset_args: tuple[str, str]
):
    """OVPhysX should resolve the heterogeneous shapes preset in either CLI order."""
    old_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0], *preset_args]
        env_cfg, _ = resolve_task_config(task_name, "rsl_rl_cfg_entry_point")
    finally:
        sys.argv = old_argv

    assert isinstance(env_cfg.scene.object.spawn, MultiAssetSpawnerCfg)
    assert len(env_cfg.scene.object.spawn.assets_cfg) == 16
