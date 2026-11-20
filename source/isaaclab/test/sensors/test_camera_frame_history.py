# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frame-history output of the camera: allocation, configuration and the default push."""

from types import MethodType, SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab.renderers import RenderBufferKind, RenderBufferSpec
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sensors.camera.camera_data import CameraData
from isaaclab.utils.tensor_processors import TensorProcessor, TensorProcessorCfg
from isaaclab.utils.warp import ProxyArray

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
    camera = SimpleNamespace(_data=data, cfg=SimpleNamespace(renderer_cfg=SimpleNamespace()))
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
    camera = SimpleNamespace(_data=data, cfg=SimpleNamespace(renderer_cfg=SimpleNamespace(provides_frame_history=True)))
    data.output["rgb"].torch[:] = 9
    push(camera, [True] * 4)
    assert data.frame_history.torch.sum() == 0


class _DoubleRGB(TensorProcessor):
    def __init__(self, cfg, data):
        super().__init__(cfg)
        self.output = ProxyArray(wp.zeros_like(data["rgb"].warp))

    def __call__(self, data, env_mask=None):
        torch.mul(data["rgb"].torch, 2, out=self.output.torch)
        return {"rgb": self.output}


@pytest.mark.parametrize("batch", [False, True])
def test_capture_history_contains_processor_output_for_single_and_batched_updates(batch):
    data = allocate(2)
    mask = wp.array([True] * 4, dtype=wp.bool, device="cpu")

    def render_camera(renderer, render_data, camera_data, step):
        camera_data.output["rgb"].torch.fill_(3)

    def render_cameras(cameras, step):
        for renderer, render_data, camera_data in cameras:
            render_camera(renderer, render_data, camera_data, step)

    sim = SimpleNamespace(
        get_physics_step_count=lambda: 0,
        render_context=SimpleNamespace(render_into_camera=render_camera, render_into_cameras=render_cameras),
    )
    camera = SimpleNamespace(
        cfg=CameraCfg(
            height=2,
            width=3,
            data_types=["rgb"],
            frame_history_length=2,
            tensor_processors=[TensorProcessorCfg(func=_DoubleRGB, inputs=["rgb"], outputs=["rgb"])],
        ),
        _sim=sim,
        _renderer=object(),
        _render_data=object(),
        _data=data,
        _is_outdated=mask,
        _env_mask_has_any=lambda env_mask: True,
        _prepare_camera=lambda env_mask: None,
        _render_outputs=dict(data.output),
        _tensor_processors=[],
    )
    camera._apply_tensor_processors = MethodType(Camera._apply_tensor_processors, camera)
    camera._push_frame_history = MethodType(Camera._push_frame_history, camera)
    if batch:
        Camera._update_buffers_batch_impl([camera])
    else:
        Camera._update_buffers_impl(camera, mask)
    assert (data.output["rgb"].torch == 6).all()
    assert (data.frame_history.torch[:, -1] == 6).all()
    assert not data.frame_history.torch[:, 0].any()
    assert data.frame_history_mask.torch.all()
