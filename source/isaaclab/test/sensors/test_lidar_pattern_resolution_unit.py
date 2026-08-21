# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from isaaclab.sensors.ray_caster.patterns import patterns, patterns_cfg

pytestmark = pytest.mark.unit


def test_lidar_nondivisible_fov_preserves_horizontal_resolution():
    """A partial FOV must not shrink the requested spacing to force the final endpoint."""
    cfg = patterns_cfg.LidarPatternCfg(
        channels=1,
        vertical_fov_range=(0.0, 0.0),
        horizontal_fov_range=(0.0, 100.0),
        horizontal_res=30.0,
    )

    _, directions = patterns.lidar_pattern(cfg, "cpu")
    angles = torch.rad2deg(torch.atan2(directions[:, 1], directions[:, 0]))

    torch.testing.assert_close(angles, torch.tensor([0.0, 30.0, 60.0, 90.0]), atol=1.0e-5, rtol=0.0)
    torch.testing.assert_close(
        torch.diff(angles), torch.full((3,), 30.0), atol=1.0e-5, rtol=0.0
    )


def test_lidar_rejects_nonpositive_horizontal_resolution():
    cfg = patterns_cfg.LidarPatternCfg(
        channels=1,
        vertical_fov_range=(0.0, 0.0),
        horizontal_fov_range=(0.0, 90.0),
        horizontal_res=0.0,
    )

    with pytest.raises(ValueError, match="Horizontal resolution must be greater than 0"):
        patterns.lidar_pattern(cfg, "cpu")
