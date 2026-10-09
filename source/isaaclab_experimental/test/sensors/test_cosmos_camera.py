# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cosmos on compatible cameras: canvas choice, preserved view, and published pixels."""

import pytest
import torch
from isaaclab_experimental.cosmos import CosmosModelCfg, CosmosTransferModifierCfg, cosmos_camera
from isaaclab_experimental.image_transfer import center_crop_resize

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

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
    camera = cosmos_camera(_camera(width, height), CosmosModelCfg(modality="depth"), near=0.2, far=1.5)
    chain = camera.modifiers["distance_to_image_plane"]

    assert (camera.width, camera.height) == canvas
    assert camera.render_data_types() == ["distance_to_image_plane"]
    assert camera.modifier_outputs() == {"distance_to_image_plane": "rgb"}
    assert chain[0].params == {"near": 0.2, "far": 1.5} and isinstance(chain[1], CosmosTransferModifierCfg)
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
        published = chain[2].func(generated, **chain[2].params)
    assert published.shape == (2, height, width, 3)
    assert published.dtype == torch.uint8 and published.device == generated.device
    assert published.is_contiguous()
    assert torch.equal(published, colors.expand(2, height, width, 3))


def test_cosmos_camera_rejects_cameras_it_cannot_reproduce():
    with pytest.raises(ValueError, match="only 'rgb'"):
        cosmos_camera(_camera(64, 64).replace(data_types=["rgb", "depth"]), CosmosModelCfg(modality="depth"))
    with pytest.raises(ValueError, match="already has modifiers"):
        cosmos_camera(_camera(64, 64, modifiers={"rgb": []}), CosmosModelCfg(modality="depth"))
    with pytest.raises(ValueError, match="depth, edge, or blur"):
        cosmos_camera(_camera(64, 64), CosmosModelCfg(modality="seg"))
    with pytest.raises(ValueError, match="fisheye"):
        cosmos_camera(_camera(64, 64, spawn=sim_utils.FisheyeCameraCfg()), CosmosModelCfg(modality="depth"))
    with pytest.raises(ValueError, match="uint8"):
        center_crop_resize(torch.zeros(1, 8, 8, 3), width=4, height=4)


def test_a_blur_camera_sends_its_rgb_for_the_service_to_blur():
    """Blur control is the camera's RGB; the service applies the Framework's own blur filter."""
    camera = cosmos_camera(_camera(640, 640), CosmosModelCfg(modality="blur"))
    chain = camera.modifiers["rgb"]

    assert camera.render_data_types() == ["rgb"]
    assert camera.modifier_outputs() == {"rgb": "rgb"}
    assert len(chain) == 1 and isinstance(chain[0], CosmosTransferModifierCfg)
