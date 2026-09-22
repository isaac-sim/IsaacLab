# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the camera observation terms.

A minimal fake sensor stands in for the camera so the ring-buffer and channel-stacking logic of
:class:`~isaaclab.envs.mdp.observations.stacked_image` runs through the real :func:`image` term
without a Kit launch.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs.mdp.observations import image, image_features, stacked_image

pytestmark = pytest.mark.unit

NUM_ENVS = 4
FRAME_SHAPE = (NUM_ENVS, 8, 8, 3)
CHANNELS = FRAME_SHAPE[-1]


def _frame(value: float, dtype: torch.dtype = torch.float32, device: str = "cpu") -> torch.Tensor:
    """Build a constant-valued ``(N, H, W, C)`` frame."""
    return torch.full(FRAME_SHAPE, value, dtype=dtype, device=device)


def _make_env(camera_output: torch.Tensor, device: str = "cpu") -> SimpleNamespace:
    """Fake env whose ``tiled_camera`` sensor returns ``camera_output`` for the ``rgb`` data type."""
    sensor = SimpleNamespace(data=SimpleNamespace(output={"rgb": camera_output}))
    return SimpleNamespace(scene=SimpleNamespace(sensors={"tiled_camera": sensor}), num_envs=NUM_ENVS, device=device)


def _set_frame(env: SimpleNamespace, frame: torch.Tensor) -> None:
    env.scene.sensors["tiled_camera"].data.output["rgb"] = frame


def _make_term(frame_stack: int, env: SimpleNamespace) -> stacked_image:
    return stacked_image(SimpleNamespace(params={"frame_stack": frame_stack}), env)


def _slots(stacked: torch.Tensor) -> list[torch.Tensor]:
    """Split a channel-stacked tensor into its per-frame slots, oldest first."""
    return list(stacked.split(CHANNELS, dim=-1))


def test_invalid_frame_stack_raises():
    with pytest.raises(ValueError, match="frame_stack must be >= 1"):
        _make_term(0, _make_env(_frame(0)))


def test_frame_stack_one_returns_the_single_frame():
    env = _make_env(_frame(42))
    out = _make_term(1, env)(env, normalize=False)
    assert torch.equal(out, _frame(42))


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"))],
)
def test_history_fills_shifts_and_resets(device):
    """The ring warms up with the first frame, shifts oldest to newest, and resets per environment."""
    env = _make_env(_frame(10, device=device), device=device)
    term = _make_term(3, env)

    # the first call fills every slot with the current frame
    out = term(env, normalize=False)
    assert out.shape == (*FRAME_SHAPE[:-1], 3 * CHANNELS)
    assert out.device.type == device
    assert all(torch.equal(slot, _frame(10, device=device)) for slot in _slots(out))

    # later frames enter at the newest slot, also well past the ring length
    for value in range(11, 21):
        _set_frame(env, _frame(value, device=device))
        out = term(env, normalize=False)
    assert [slot[0, 0, 0, 0].item() for slot in _slots(out)] == [18, 19, 20]

    # a partial reset re-fills only the reset environment
    term.reset(torch.tensor([0], device=device))
    _set_frame(env, _frame(9, device=device))
    out = term(env, normalize=False)
    assert [slot[0, 0, 0, 0].item() for slot in _slots(out)] == [9, 9, 9]
    assert [slot[1, 0, 0, 0].item() for slot in _slots(out)] == [19, 20, 9]

    # a full reset re-fills every environment
    term.reset()
    _set_frame(env, _frame(50, device=device))
    out = term(env, normalize=False)
    assert all(torch.equal(slot, _frame(50, device=device)) for slot in _slots(out))


def test_rgb_frames_are_buffered_raw_and_normalized_per_frame():
    """uint8 RGB frames stay uint8 in the ring; normalization matches the per-frame math and never aliases."""
    frames = [torch.randint(0, 255, FRAME_SHAPE, dtype=torch.uint8) for _ in range(2)]
    env = _make_env(frames[0])
    term = _make_term(2, env)

    out_old = term(env, normalize=True, data_type="rgb")
    _set_frame(env, frames[1])
    out = term(env, normalize=True, data_type="rgb")

    assert term._buffer._buffer.dtype == torch.uint8
    for slot, frame in zip(_slots(out), frames, strict=True):
        scaled = frame.float() / 255.0
        torch.testing.assert_close(slot, scaled - scaled.mean(dim=(1, 2), keepdim=True))
    # consecutive outputs must not share storage: the trainer keeps the previous observation alive
    assert out_old.data_ptr() != out.data_ptr()


@pytest.mark.parametrize("clone", [False, True])
def test_image_clone_flag_controls_storage_sharing(clone):
    camera_buf = torch.randint(0, 255, FRAME_SHAPE, dtype=torch.uint8)
    env = _make_env(camera_buf)
    out = image(env, sensor_cfg=SimpleNamespace(name="tiled_camera"), normalize=False, clone=clone)
    assert (out.data_ptr() == camera_buf.data_ptr()) is not clone


def test_image_features_flattens_encoder_output():
    """Feature extractors return a flat observation after the environment batch dimension."""
    env = _make_env(_frame(0, dtype=torch.uint8))
    term = image_features.__new__(image_features)
    term._model = object()
    term._inference_fn = lambda *_args, **_kwargs: torch.arange(NUM_ENVS * 6 * 8).reshape(NUM_ENVS, 6, 8)

    assert term(env).shape == (NUM_ENVS, 48)
