# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavioral tests for the unified dexterous Lift and Reorient tasks."""

import torch

from isaaclab_tasks.core.lift import mdp


def test_camera_normalization_is_stationary() -> None:
    """RGB and depth normalization must not depend on per-frame statistics."""
    rgb = torch.tensor([0.0, 127.5, 255.0])
    depth = torch.tensor([0.0, 2.0])

    assert torch.allclose(mdp.vision_camera._rgb_norm(None, rgb), torch.tensor([-0.5, 0.0, 0.5]))
    assert torch.allclose(mdp.vision_camera._depth_norm(None, depth), torch.tanh(depth / 2) - 0.5)
