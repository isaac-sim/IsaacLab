# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.scene_data as scene_data


def test_scene_data_requirement_mapping():
    assert scene_data.REQUIRES_STAGE_AND_MODEL == {
        "kit": (True, False),
        "newton_gl": (False, True),
        "newton": (False, True),
        "newton_rtx": (False, True),
        "rerun": (False, True),
        "viser": (False, True),
        "isaac_rtx": (True, False),
        "newton_warp": (False, True),
        "ovrtx": (True, True),
    }
    for name in ("TYPES", "REQUIRES_STAGE", "REQUIRES_MODEL"):
        assert not hasattr(scene_data, name)
