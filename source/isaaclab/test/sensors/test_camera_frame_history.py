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
from isaaclab.test.utils import test_devices

SPECS = {RenderBufferKind.RGBA: RenderBufferSpec(4, wp.uint8), RenderBufferKind.RGB: RenderBufferSpec(3, wp.uint8)}


def allocate(history_length, device="cpu"):
    data = CameraData.allocate(["rgb"], 2, 3, 4, device, SPECS, history_length=history_length)
    data.create_buffers(4, device)
    return data


def test_history_shape_resolves_to_five_dimensions_in_torch():
    data = allocate(5)
    assert data.frame_history.torch.shape == (4, 5, 2, 3, 3)
    assert data.frame_history.torch.dtype == torch.uint8
    assert data.frame_history_mask.torch.shape == (4,)


@pytest.mark.parametrize("data_types,length,error", [(["depth"], 2, "requires 'rgb'"), (["rgb"], -2, r"0 \(off\)")])
def test_history_requires_rgb_and_a_non_negative_length(data_types, length, error):
    with pytest.raises(ValueError, match=error):
        CameraCfg(height=2, width=3, data_types=data_types, frame_history_length=length).validate_config()


def fake_camera(data, provides_frame_history=False):
    """The parts of a camera that the history push reads."""
    camera = SimpleNamespace(
        _data=data,
        _renderer=SimpleNamespace(provides_frame_history=provides_frame_history),
        cfg=SimpleNamespace(renderer_cfg=None),
    )
    Camera._create_frame_history_state(camera)
    return camera


def push(camera, mask, device="cpu"):
    Camera._push_frame_history(camera, wp.array(mask, dtype=wp.bool, device=device))


@pytest.mark.parametrize("device", test_devices())
def test_default_push_appends_the_capture_of_updated_environments_only(device):
    data = allocate(3, device)
    camera = fake_camera(data)
    rgb = data.output["rgb"].torch
    mask = wp.array([True, True, False, False], dtype=wp.bool, device=device)
    for value in (1, 2):
        rgb[:] = value
        if device.startswith("cuda"):
            # The push runs on every camera update, so it must not wait for the GPU.
            torch.cuda.set_sync_debug_mode("error")
        try:
            Camera._push_frame_history(camera, mask)
        finally:
            if device.startswith("cuda"):
                torch.cuda.set_sync_debug_mode("default")
    history = data.frame_history.torch
    # The first capture fills the whole history, so it never starts with empty frames.
    assert history[0, :, 0, 0, 0].tolist() == [1, 1, 2]
    assert history[2].sum() == 0
    assert data.frame_history_mask.torch.tolist() == [True, True, False, False]
    # After environment 0 resets, its first capture replaces every frame of the ended episode.
    camera._history_restart[0] = True
    rgb[:] = 5
    Camera._push_frame_history(camera, mask)
    assert history[0, :, 0, 0, 0].tolist() == [5, 5, 5] and history[1, :, 0, 0, 0].tolist() == [1, 2, 5]


def test_a_renderer_that_provides_the_history_is_left_alone():
    data = allocate(3)
    camera = fake_camera(data, provides_frame_history=True)
    data.output["rgb"].torch[:] = 9
    push(camera, [True] * 4)
    assert data.frame_history.torch.sum() == 0
