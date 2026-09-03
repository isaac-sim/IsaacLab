# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim-free tests for the Cartpole camera observation term.

The term must hand the policy a float32 tensor for every camera data type. Segmentation is the
interesting case: it is ``uint8`` RGBA when colorized and ``int32`` label ids when not, and the
feature extractor's first convolution rejects both integer dtypes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import ObservationTermCfg, SceneEntityCfg

from isaaclab_tasks.core.cartpole.mdp.observations import CameraImageStack

pytestmark = pytest.mark.unit


@pytest.fixture(params=["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"])
def device(request):
    return request.param


def _make_env(camera_output: dict[str, torch.Tensor], frame_stack: int, device: str) -> SimpleNamespace:
    """Build the minimal environment stub the observation term reads from."""
    camera = SimpleNamespace(data=SimpleNamespace(output=camera_output))
    return SimpleNamespace(
        cfg=SimpleNamespace(frame_stack=frame_stack),
        num_envs=next(iter(camera_output.values())).shape[0],
        device=device,
        scene=SimpleNamespace(sensors={"tiled_camera": camera}),
    )


@pytest.mark.parametrize("frame_stack", [1, 2])
@pytest.mark.parametrize(
    "dtype,num_channels",
    [(torch.uint8, 4), (torch.int32, 1)],
    ids=["colorized_uint8_rgba", "non_colorized_int32_labels"],
)
def test_semantic_segmentation_observation_is_float32(device, frame_stack, dtype, num_channels):
    """Segmentation observations are float32 regardless of the renderer's ``colorize`` setting."""
    torch.manual_seed(0)
    images = torch.randint(0, 5, (2, 8, 8, num_channels), dtype=dtype, device=device)
    env = _make_env({"semantic_segmentation": images}, frame_stack, device)
    term = CameraImageStack(ObservationTermCfg(func=CameraImageStack), env)

    observation = term(env, SceneEntityCfg("tiled_camera"), "semantic_segmentation")

    assert observation.dtype == torch.float32
    assert observation.shape == (2, frame_stack * num_channels, 8, 8)


def test_colorized_segmentation_matches_rgb_normalization(device):
    """Colorized segmentation gets the same ``(x / 255) - per-image mean`` treatment as RGB."""
    torch.manual_seed(0)
    images = torch.randint(0, 255, (2, 8, 8, 4), dtype=torch.uint8, device=device)
    env = _make_env({"semantic_segmentation": images}, frame_stack=1, device=device)
    term = CameraImageStack(ObservationTermCfg(func=CameraImageStack), env)

    observation = term(env, SceneEntityCfg("tiled_camera"), "semantic_segmentation")

    expected = images.float() / 255.0
    expected = expected - torch.mean(expected, dim=(1, 2), keepdim=True)
    torch.testing.assert_close(observation, expected.permute(0, 3, 1, 2), atol=1e-5, rtol=1e-5)


def test_non_colorized_segmentation_preserves_label_ids(device):
    """Label ids carry no scale, so the int32 map is only cast, never rescaled."""
    images = torch.arange(2 * 8 * 8, dtype=torch.int32, device=device).reshape(2, 8, 8, 1) % 5
    env = _make_env({"semantic_segmentation": images}, frame_stack=1, device=device)
    term = CameraImageStack(ObservationTermCfg(func=CameraImageStack), env)

    observation = term(env, SceneEntityCfg("tiled_camera"), "semantic_segmentation")

    torch.testing.assert_close(observation, images.to(torch.float32).permute(0, 3, 1, 2))
