# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frame-history output of the camera: allocation, configuration and the default push."""

from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab.renderers import RenderBufferKind, RenderBufferSpec
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sensors.camera.camera_data import CameraData

SPECS = {RenderBufferKind.RGBA: RenderBufferSpec(4, wp.uint8), RenderBufferKind.RGB: RenderBufferSpec(3, wp.uint8)}


def allocate(history_length):
    data = CameraData.allocate(["rgb"], 2, 3, 4, "cpu", SPECS, history_length=history_length)
    data.create_buffers(4, "cpu")
    return data


def test_history_shape_resolves_to_five_dimensions_in_torch():
    data = allocate(5)
    assert data.frame_history.torch.shape == (4, 5, 2, 3, 3)
    assert data.frame_history.torch.dtype == torch.uint8
    assert data.frame_history_mask.torch.shape == (4,)


def test_history_requires_rgb():
    with pytest.raises(ValueError, match="requires 'rgb'"):
        CameraCfg(height=2, width=3, data_types=["depth"], frame_history_length=2).validate_config()


def push(camera, mask):
    Camera._push_frame_history(camera, wp.array(mask, dtype=wp.bool, device="cpu"))


def test_default_push_appends_the_capture_of_updated_environments_only():
    data = allocate(3)
    camera = SimpleNamespace(_data=data, _renderer=SimpleNamespace(provides_frame_history=False))
    rgb = data.output["rgb"].torch
    for value in (1, 2):
        rgb[:] = value
        push(camera, [True, True, False, False])
    history = data.frame_history.torch
    assert history[0, :, 0, 0, 0].tolist() == [0, 1, 2]
    assert history[2].sum() == 0
    assert data.frame_history_mask.torch.tolist() == [True, True, False, False]


def test_a_renderer_that_provides_the_history_is_left_alone():
    data = allocate(3)
    camera = SimpleNamespace(_data=data, _renderer=SimpleNamespace(provides_frame_history=True))
    data.output["rgb"].torch[:] = 9
    push(camera, [True] * 4)
    assert data.frame_history.torch.sum() == 0
