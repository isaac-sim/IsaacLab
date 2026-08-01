# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for DexSuite Kuka Allegro OVPhysX object preset compatibility."""

import sys

import pytest

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config


@pytest.mark.parametrize(
    "task_name",
    [
        "Isaac-Lift-KukaAllegro",
        "Isaac-Reorient-KukaAllegro",
    ],
)
@pytest.mark.parametrize(
    "preset_args",
    [
        ("physics=ovphysx", "presets=shapes"),
        ("presets=shapes", "physics=ovphysx"),
    ],
)
def test_ovphysx_rejects_heterogeneous_shapes_independent_of_argument_order(
    task_name: str, preset_args: tuple[str, str]
):
    """OVPhysX should reject the unsupported heterogeneous shapes preset in either CLI order."""
    old_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0], *preset_args]
        with pytest.raises(
            ValueError,
            match=r"Conflicting global presets: .* both define preset for 'env\.scene\.object\.spawn'",
        ):
            resolve_task_config(task_name, "rsl_rl_cfg_entry_point")
    finally:
        sys.argv = old_argv
