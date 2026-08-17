# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.scene_data as scene_data


def test_scene_data_requirements_use_one_mapping():
    assert isinstance(scene_data.REQUIRES_STAGE_AND_MODEL, dict)
    assert not any(hasattr(scene_data, name) for name in ("TYPES", "REQUIRES_STAGE", "REQUIRES_MODEL"))
