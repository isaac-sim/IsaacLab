# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim-free tests for the Cartpole camera observation term.

Segmentation output is ``uint8`` RGBA when colorized and ``int32`` label ids when not. Either
integer dtype crashes the feature extractor's first convolution, so the term must return float32
for both. ``frame_stack`` is parametrized because the stacking path defers normalization.
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


def _observe(images: torch.Tensor, frame_stack: int, device: str) -> torch.Tensor:
    """Run the observation term over ``images`` using a minimal environment stub."""
    camera = SimpleNamespace(data=SimpleNamespace(output={"semantic_segmentation": images}))
    env = SimpleNamespace(
        cfg=SimpleNamespace(frame_stack=frame_stack),
        num_envs=images.shape[0],
        device=device,
        scene=SimpleNamespace(sensors={"tiled_camera": camera}),
    )
    term = CameraImageStack(ObservationTermCfg(func=CameraImageStack), env)
    return term(env, SceneEntityCfg("tiled_camera"), "semantic_segmentation")


def _to_expected_layout(images: torch.Tensor, frame_stack: int) -> torch.Tensor:
    """Convert BHWC to the channel-first layout, repeated as the ring buffer fills on first append."""
    return images.permute(0, 3, 1, 2).repeat(1, frame_stack, 1, 1)


@pytest.mark.parametrize("frame_stack", [1, 2])
def test_colorized_segmentation_is_normalized_like_rgb(device, frame_stack):
    """Colorized uint8 RGBA segmentation gets the same ``(x / 255) - per-image mean`` as RGB."""
    torch.manual_seed(0)
    images = torch.randint(0, 255, (2, 8, 8, 4), dtype=torch.uint8, device=device)

    observation = _observe(images, frame_stack, device)

    expected = images.float() / 255.0
    expected = expected - torch.mean(expected, dim=(1, 2), keepdim=True)
    assert observation.dtype == torch.float32
    torch.testing.assert_close(observation, _to_expected_layout(expected, frame_stack), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("frame_stack", [1, 2])
def test_non_colorized_segmentation_is_cast_to_float(device, frame_stack):
    """Non-colorized int32 label ids are cast to float32 and, carrying no scale, left unrescaled."""
    images = torch.arange(2 * 8 * 8, dtype=torch.int32, device=device).reshape(2, 8, 8, 1) % 5

    observation = _observe(images, frame_stack, device)

    assert observation.dtype == torch.float32
    torch.testing.assert_close(observation, _to_expected_layout(images.float(), frame_stack))
