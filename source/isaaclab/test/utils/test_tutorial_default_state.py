# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_add_new_robot_wave_command_does_not_alias_default_joint_state():
    repo_root = Path(__file__).parents[4]
    tutorial = repo_root / "scripts" / "tutorials" / "01_assets" / "add_new_robot.py"
    source = tutorial.read_text(encoding="utf-8")

    assert 'wave_action = scene["Dofbot"].data.default_joint_pos.torch.clone()' in source
    assert 'wave_action = scene["Dofbot"].data.default_joint_pos.torch\n' not in source
