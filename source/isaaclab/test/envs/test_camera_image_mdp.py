# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the camera image observation terms in :mod:`isaaclab.envs.mdp.observations`.

The camera is a stand-in whose ``data.output`` entries are :class:`ProxyArray` views over torch tensors,
like real sensors, so the tests exercise the terms without a Kit launch. Writing into a camera tensor
in place simulates a new rendered frame.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import warp as wp

pytestmark = pytest.mark.unit

from isaaclab.envs.mdp.observations import (
    image_depth,
    image_features,
    image_rgb,
    image_segmentation,
    stacked_image,
)
from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.warp import ProxyArray

NUM_ENVS = 4
HEIGHT = 8
WIDTH = 8
CHANNELS = 3
SENSOR_CFG = SceneEntityCfg("tiled_camera")


def _make_env(outputs: dict[str, torch.Tensor], device: str = "cpu") -> SimpleNamespace:
    """Mock env exposing ``env.scene.sensors["tiled_camera"].data.output`` over ``outputs``."""
    camera = SimpleNamespace(
        data=SimpleNamespace(output={name: ProxyArray(wp.from_torch(buf)) for name, buf in outputs.items()})
    )
    return SimpleNamespace(num_envs=NUM_ENVS, device=device, scene=SimpleNamespace(sensors={"tiled_camera": camera}))


def _make_term(term_cls, env: SimpleNamespace, **params):
    """Construct a class term and a bound call using the same params, as the observation manager does."""
    params = {"sensor_cfg": SENSOR_CFG, **params}
    term = term_cls(ObservationTermCfg(func=term_cls, params=params), env)
    return term, lambda: term(env, **params)


def _rgb_reference(frame: torch.Tensor) -> torch.Tensor:
    x = frame.float() / 255.0
    return x - torch.mean(x, dim=(1, 2), keepdim=True)


def _random_rgb() -> torch.Tensor:
    return torch.randint(0, 255, (NUM_ENVS, HEIGHT, WIDTH, CHANNELS), dtype=torch.uint8)


class TestFrameStacking:
    """History, reset and ownership behavior shared by every camera image term."""

    def test_oldest_to_newest_channel_order(self):
        """K=3 with three distinct frames produces oldest to newest along the channel dim."""
        camera = torch.zeros((NUM_ENVS, HEIGHT, WIDTH, CHANNELS), dtype=torch.uint8)
        env = _make_env({"rgb": camera})
        _, observe = _make_term(image_rgb, env, normalize=False, frame_stack=3)
        for value in (10, 20, 30):
            camera.fill_(value)
            out = observe()
        assert out.shape == (NUM_ENVS, HEIGHT, WIDTH, CHANNELS * 3)
        for slot, value in enumerate((10, 20, 30)):
            assert torch.all(out[..., slot * CHANNELS : (slot + 1) * CHANNELS] == value)

    def test_reset_partial_envs_preserves_others(self):
        """Resetting env 0 refills only env 0's history with the next frame."""
        camera = torch.full((NUM_ENVS, HEIGHT, WIDTH, CHANNELS), 1, dtype=torch.uint8)
        env = _make_env({"rgb": camera})
        term, observe = _make_term(image_rgb, env, normalize=False, frame_stack=2)
        observe()
        camera.fill_(2)
        observe()
        term.reset(torch.tensor([0]))
        camera.fill_(9)
        out = observe()
        assert torch.all(out[0] == 9)
        assert torch.all(out[1, ..., :CHANNELS] == 2)
        assert torch.all(out[1, ..., CHANNELS:] == 9)

    def test_invalid_frame_stack_raises(self):
        with pytest.raises(ValueError, match="frame_stack must be >= 1"):
            _make_term(image_rgb, _make_env({"rgb": _random_rgb()}), frame_stack=0)

    @pytest.mark.parametrize("frame_stack", [1, 2])
    def test_rgb_normalize_matches_per_frame_math(self, frame_stack):
        """Normalizing after stacking uint8 frames equals normalizing each frame independently."""
        camera = _random_rgb()
        env = _make_env({"rgb": camera})
        _, observe = _make_term(image_rgb, env, frame_stack=frame_stack)
        first = camera.clone()
        observe()
        camera.copy_(_random_rgb())
        out = observe()
        expected = [_rgb_reference(camera)] if frame_stack == 1 else [_rgb_reference(first), _rgb_reference(camera)]
        torch.testing.assert_close(out, torch.cat(expected, dim=-1), atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("frame_stack", [1, 2])
    def test_returns_storage_independent_of_camera_and_previous_step(self, frame_stack):
        """The trainer holds the previous observation across a step; it must not be overwritten."""
        camera = _random_rgb()
        env = _make_env({"rgb": camera})
        _, observe = _make_term(image_rgb, env, normalize=False, frame_stack=frame_stack)
        out_a = observe()
        snapshot = out_a.clone()
        camera.copy_(_random_rgb())
        out_b = observe()
        assert out_a.data_ptr() not in (out_b.data_ptr(), camera.data_ptr())
        torch.testing.assert_close(out_a, snapshot)

    def test_channel_first_stack(self):
        camera = _random_rgb()
        env = _make_env({"rgb": camera})
        _, observe = _make_term(image_rgb, env, channel_first=True, frame_stack=2)
        out = observe()
        expected = _rgb_reference(camera).permute(0, 3, 1, 2).repeat(1, 2, 1, 1)
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available in this env")
    def test_buffer_on_cuda(self):
        camera = _random_rgb().cuda()
        env = _make_env({"rgb": camera}, device="cuda")
        _, observe = _make_term(image_rgb, env, frame_stack=2)
        out = observe()
        assert out.device.type == "cuda"
        torch.testing.assert_close(out, torch.cat([_rgb_reference(camera)] * 2, dim=-1), atol=1e-5, rtol=1e-5)


class TestModalityTerms:
    """Per-modality reading, validation and normalization."""

    def test_rgb_drops_alpha_and_accepts_constant_mean(self):
        """Albedo's unused alpha channel is dropped and a constant mean replaces the per-image mean."""
        albedo = torch.randint(0, 255, (NUM_ENVS, HEIGHT, WIDTH, 4), dtype=torch.uint8)
        env = _make_env({"albedo": albedo})
        _, observe = _make_term(image_rgb, env, data_type="albedo", mean=0.5)
        torch.testing.assert_close(observe(), albedo[..., :3].float() / 255.0 - 0.5, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize(("term_cls", "data_type"), [(image_rgb, "depth"), (image_depth, "rgb")])
    def test_rejects_other_modality(self, term_cls, data_type):
        with pytest.raises(ValueError, match="does not support camera data type"):
            _make_term(term_cls, _make_env({data_type: _random_rgb()}), data_type=data_type)

    @pytest.mark.parametrize("frame_stack", [1, 2])
    def test_depth_replaces_invalid_and_rescales(self, frame_stack):
        depth = torch.tensor([0.0, 5.0, 20.0, float("inf")]).repeat(NUM_ENVS, HEIGHT, 2).unsqueeze(-1)
        env = _make_env({"distance_to_image_plane": depth})
        _, observe = _make_term(image_depth, env, invalid_value=10.0, max_depth=10.0, frame_stack=frame_stack)
        expected = torch.tensor([0.0, 0.5, 1.0, 1.0]).repeat(NUM_ENVS, HEIGHT, 2).unsqueeze(-1)
        torch.testing.assert_close(observe(), expected.repeat(1, 1, 1, frame_stack))
        assert torch.isinf(depth).any(), "the camera buffer must not be modified"

    @pytest.mark.parametrize("frame_stack", [1, 2])
    def test_colorized_segmentation_is_normalized_like_rgb(self, frame_stack):
        """Colorized uint8 RGBA segmentation keeps its four channels and is scaled like color."""
        seg = torch.randint(0, 255, (NUM_ENVS, HEIGHT, WIDTH, 4), dtype=torch.uint8)
        env = _make_env({"semantic_segmentation": seg})
        _, observe = _make_term(image_segmentation, env, channel_first=True, frame_stack=frame_stack)
        expected = _rgb_reference(seg).permute(0, 3, 1, 2).repeat(1, frame_stack, 1, 1)
        torch.testing.assert_close(observe(), expected, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("frame_stack", [1, 2])
    def test_label_segmentation_is_cast_to_float(self, frame_stack):
        """int32 label ids carry no scale: they are cast to float32 and not rescaled."""
        seg = torch.arange(NUM_ENVS * HEIGHT * WIDTH, dtype=torch.int32).reshape(NUM_ENVS, HEIGHT, WIDTH, 1) % 5
        env = _make_env({"semantic_segmentation": seg})
        _, observe = _make_term(image_segmentation, env, channel_first=True, frame_stack=frame_stack)
        out = observe()
        assert out.dtype == torch.float32
        torch.testing.assert_close(out, seg.float().permute(0, 3, 1, 2).repeat(1, frame_stack, 1, 1))


class TestDeprecatedTerms:
    """``stacked_image`` keeps working and points to the per-modality terms."""

    def test_stacked_image_forwards(self):
        camera = _random_rgb()
        env = _make_env({"rgb": camera})
        with pytest.deprecated_call():
            _, observe = _make_term(stacked_image, env, frame_stack=2)
        torch.testing.assert_close(observe(), torch.cat([_rgb_reference(camera)] * 2, dim=-1), atol=1e-5, rtol=1e-5)


def test_image_features_flattens_encoder_output():
    """Feature extractors return a flat observation after the environment batch dimension."""
    env = _make_env({"rgb": _random_rgb()})
    term = image_features.__new__(image_features)
    term._model = object()
    term._inference_fn = lambda *_args, **_kwargs: torch.arange(NUM_ENVS * 6 * 8).reshape(NUM_ENVS, 6, 8)
    assert term(env, sensor_cfg=SENSOR_CFG).shape == (NUM_ENVS, 48)
