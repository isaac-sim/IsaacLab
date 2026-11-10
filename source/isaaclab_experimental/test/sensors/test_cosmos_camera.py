# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cosmos on compatible cameras: canvas choice, preserved view, and published pixels."""

import pytest
import torch
from isaaclab_experimental.cosmos import CosmosModelCfg, CosmosTransferProcessorCfg, cosmos_camera
from isaaclab_experimental.image_transfer import FunctionProcessorCfg

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.camera import CAPTURE_ENV_MASK_NAME

pytestmark = pytest.mark.unit


def _camera(width, height, **kwargs):
    return CameraCfg(
        prim_path="/World/Camera",
        spawn=kwargs.pop("spawn", sim_utils.PinholeCameraCfg(horizontal_aperture=20.0)),
        data_types=["rgb"],
        width=width,
        height=height,
        **kwargs,
    )


@pytest.mark.parametrize(
    "size,canvas,bounds",
    [
        ((64, 64), (640, 640), (0, 0, 640, 640)),
        ((160, 120), (736, 544), (5, 0, 730, 544)),
        ((120, 160), (544, 736), (0, 5, 544, 730)),
        ((832, 480), (832, 480), (0, 0, 832, 480)),
    ],
)
def test_cosmos_camera_renders_a_canvas_with_the_same_view_and_publishes_the_original_size(size, canvas, bounds):
    """The canvas keeps the camera's aspect ratio; the published crop sees what the camera saw."""
    width, height = size
    camera = cosmos_camera(_camera(width, height), CosmosModelCfg(), near=0.2, far=1.5)
    chain = camera.tensor_processors

    assert (camera.width, camera.height) == canvas
    # The renderer produces depth; the processors produce the published rgb.
    assert chain[0].inputs == ["distance_to_image_plane"] and chain[-1].outputs == ["rgb"]
    assert isinstance(chain[0], FunctionProcessorCfg) and chain[0].params == {"near": 0.2, "far": 1.5}
    assert isinstance(chain[1], CosmosTransferProcessorCfg) and chain[1].inputs == ["control", CAPTURE_ENV_MASK_NAME]
    # Square pixels: apertures scale with the canvas, and the crop covers the original apertures.
    pixel = camera.spawn.horizontal_aperture / canvas[0]
    assert camera.spawn.vertical_aperture == pytest.approx(pixel * canvas[1])
    crop = min(canvas[0] / width, canvas[1] / height)
    assert pixel * width * crop == pytest.approx(20.0, rel=1 / min(canvas))
    # Two differently colored views with a white border outside the independently specified center crop.
    # Publishing a corner crop, the entire canvas, or another view's pixels changes the expected colors.
    colors = torch.tensor([[20, 40, 60], [80, 100, 120]], dtype=torch.uint8).view(2, 1, 1, 3)
    generated = torch.full((2, canvas[1], canvas[0], 3), 255, dtype=torch.uint8)
    left, top, right, bottom = bounds
    generated[:, top:bottom, left:right] = colors
    if size == canvas:
        assert len(chain) == 2
        published = generated
    else:
        published = chain[2].function(generated, **chain[2].params)
    assert published.shape == (2, height, width, 3)
    assert published.dtype == torch.uint8 and published.device == generated.device
    assert published.is_contiguous()
    assert torch.equal(published, colors.expand(2, height, width, 3))


def _first_three_channels(data: torch.Tensor) -> torch.Tensor:
    return data[..., :3]


@pytest.mark.parametrize("render_output", ["semantic_segmentation", "rgb"])
def test_cosmos_camera_takes_any_processors_that_write_the_control(render_output):
    """A control other than depth: the camera renders what the control reads, and Cosmos still publishes rgb."""
    from isaaclab.sensors.camera.camera_cfg import _render_data_types

    control = FunctionProcessorCfg(function=_first_three_channels, inputs=[render_output], outputs=["control"])
    camera = cosmos_camera(_camera(64, 64), CosmosModelCfg(), control=[control])
    chain = camera.tensor_processors

    assert chain[0].function is _first_three_channels and chain[0].inputs == [render_output]
    assert isinstance(chain[1], CosmosTransferProcessorCfg) and len(chain) == 3  # control, Cosmos, resize
    assert chain[1].inputs == ["control", CAPTURE_ENV_MASK_NAME] and chain[-1].outputs == ["rgb"]
    # The renderer produces only the control's input, including rgb when the control is made from the rendered image.
    assert _render_data_types(camera) == [render_output]

    unrelated = FunctionProcessorCfg(function=_first_three_channels, inputs=["rgb"], outputs=["mask"])
    with pytest.raises(ValueError, match="'control' buffer"):
        cosmos_camera(_camera(64, 64), CosmosModelCfg(), control=[unrelated])


def test_cosmos_camera_drops_the_isp_of_the_image_it_replaces():
    """An ISP processes the rendered rgb, which the Cosmos camera no longer renders."""
    camera = cosmos_camera(_camera(64, 64).replace(isp_cfg="ppisp"), CosmosModelCfg())
    assert camera.isp_cfg is None


def test_cosmos_camera_rejects_cameras_it_cannot_reproduce():
    with pytest.raises(ValueError, match="only 'rgb'"):
        cosmos_camera(_camera(64, 64).replace(data_types=["rgb", "depth"]), CosmosModelCfg())
    existing = FunctionProcessorCfg(function=torch.clone, inputs=["rgb"], outputs=["rgb"])
    with pytest.raises(ValueError, match="already has tensor processors"):
        cosmos_camera(_camera(64, 64, tensor_processors=[existing]), CosmosModelCfg())
    with pytest.raises(ValueError, match="fisheye"):
        cosmos_camera(_camera(64, 64, spawn=sim_utils.FisheyeCameraCfg()), CosmosModelCfg())
    # Calibrated intrinsics take precedence over the aperture the canvas widens.
    calibrated = sim_utils.PinholeCameraCfg(distortion=sim_utils.OpenCvPinholeDistortionCfg())
    with pytest.raises(ValueError, match="uncalibrated pinhole"):
        cosmos_camera(_camera(64, 64, spawn=calibrated), CosmosModelCfg())
