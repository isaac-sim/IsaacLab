# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Cartpole camera observation processing."""

from types import SimpleNamespace

import torch

from isaaclab_tasks.core.cartpole.mdp.observations import CameraImageStack


def test_camera_image_stack_concatenates_multiple_aovs() -> None:
    """Concatenate normalized color and depth outputs along the channel dimension."""
    rgb = torch.full((2, 3, 4, 3), 128, dtype=torch.uint8)
    depth = torch.full((2, 3, 4, 1), 2.0)
    camera = SimpleNamespace(data=SimpleNamespace(output={"rgb": rgb, "depth": depth}))
    env = SimpleNamespace(scene=SimpleNamespace(sensors={"camera": camera}))
    term = object.__new__(CameraImageStack)
    term._stack = None

    observation = term(env, SimpleNamespace(name="camera"), ["rgb", "depth"])

    assert observation.shape == (2, 4, 3, 4)
    assert observation.dtype == torch.float32
    torch.testing.assert_close(observation[:, :3], torch.zeros_like(observation[:, :3]))
    torch.testing.assert_close(observation[:, 3:], depth.permute(0, 3, 1, 2))
