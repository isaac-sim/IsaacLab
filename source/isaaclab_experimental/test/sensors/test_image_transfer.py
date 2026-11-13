# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Image transfer scheduling, resets, and ownership through the tensor processor interface."""

from types import SimpleNamespace

import pytest
import torch
import warp as wp
from isaaclab_experimental.image_transfer import (
    FunctionProcessorCfg,
    ImageTransferProcessorCfg,
    depth_to_control,
)
from isaaclab_experimental.image_transfer import processor as processor_module

from isaaclab.sensors.camera import CAPTURE_ENV_MASK_NAME
from isaaclab.sim import BackendCfg
from isaaclab.utils import configclass
from isaaclab.utils.warp import ProxyArray

pytestmark = pytest.mark.unit


class RecordingStream:
    """Record generation requests and return each chunk's controls inverted."""

    def __init__(self, model):
        self.model = model
        self.closed = 0

    def step(self, controls, reset_rows, seeds):
        self.model.steps.append(([chunk.clone() for chunk in controls], reset_rows, seeds))
        if self.model.fail:
            raise RuntimeError("generation failed")
        return [255 - chunk for chunk in controls]

    def close(self):
        self.closed += 1


class RecordingModel:
    def __init__(self, cfg):
        self.steps = []
        self.streams = []
        self.opened = []
        self.fail = False
        self.closed = 0

    def open_stream(self, num_views, seeds):
        self.opened.append((num_views, seeds))
        self.streams.append(RecordingStream(self))
        return self.streams[-1]

    def close(self):
        self.closed += 1


@configclass
class RecordingModelCfg(BackendCfg):
    class_type: type = RecordingModel


@pytest.fixture
def no_sim(monkeypatch):
    monkeypatch.setattr(processor_module, "SimulationContext", SimpleNamespace(instance=lambda: None))


def _controls(value):
    return torch.full((2, 2, 3, 3), value, dtype=torch.uint8)


def _buffers(controls, mask=None):
    """Camera buffers holding ``controls`` and, when given, the capture mask."""
    buffers = {"control": ProxyArray(wp.from_torch(controls))}
    if mask is not None:
        buffers[CAPTURE_ENV_MASK_NAME] = ProxyArray(wp.array(mask, dtype=wp.bool, device="cpu"))
    return buffers


def _processor(**kwargs):
    cfg = ImageTransferProcessorCfg(backend=RecordingModelCfg(), seed=7, **kwargs)
    processor = cfg.func(cfg, _buffers(_controls(0)))
    return processor, processor._owned_model


def _rgb(processor, value, mask=None):
    return processor(_buffers(_controls(value), mask))["rgb"].torch


def test_chunks_follow_the_configured_cadence_and_hold_between_chunks(no_sim):
    """The first chunk uses initial_frames, later chunks update_frames; queued controls are owned copies."""
    processor, model = _processor(initial_frames=1, update_frames=2)
    controls = _controls(10)
    buffers = _buffers(controls)

    output = processor(buffers)["rgb"]
    controls.fill_(20)
    assert processor(buffers)["rgb"] is output
    assert len(model.steps) == 1
    assert torch.equal(output.torch, _controls(245))
    controls.fill_(30)
    processor(buffers)

    assert model.opened == [(2, (7, 8))]
    assert [tuple(chunk.shape) for chunk in model.steps[1][0]] == [(2, 2, 3, 3)] * 2
    torch.testing.assert_close(model.steps[1][0][0][:, 0, 0, 0], torch.tensor([20, 30], dtype=torch.uint8))
    assert torch.equal(output.torch, _controls(225))


def test_reset_restarts_only_the_selected_view_with_a_new_seed(no_sim):
    processor, model = _processor(initial_frames=1, update_frames=1)
    _rgb(processor, 10)

    processor.reset(torch.tensor([1]))
    assert torch.equal(processor._output[1], torch.zeros(2, 3, 3, dtype=torch.uint8))
    assert torch.equal(processor._output[0], torch.full((2, 3, 3), 245, dtype=torch.uint8))
    _rgb(processor, 10)

    # View 1 starts again from its seed 8, advanced by the number of views for its second episode.
    assert model.steps[-1][1:] == ((1,), (10,))


def test_a_capture_queues_controls_only_for_the_views_in_env_mask(no_sim):
    """After independent resets, environments capture at different steps; a view a capture skipped queues nothing.

    Queues hold at most one chunk, so a full view that a capture skips must not block the view that completes it.
    """
    processor, model = _processor(initial_frames=1, update_frames=2, max_pending_frames=2)
    _rgb(processor, 1, [True, True])

    # The two views now capture on alternate calls; each call's data holds a stale render for the other view.
    for value, mask in ((2, [True, False]), (3, [False, True]), (4, [True, False]), (5, [False, True])):
        _rgb(processor, value, mask)

    assert len(model.steps) == 2
    chunks = model.steps[1][0]
    assert chunks[0][:, 0, 0, 0].tolist() == [2, 4] and chunks[1][:, 0, 0, 0].tolist() == [3, 5]
    assert [len(queue) for queue in processor._pending] == [0, 0]


def test_a_failed_generation_publishes_nothing_and_is_not_retried(no_sim):
    processor, model = _processor()
    model.fail = True

    with pytest.raises(RuntimeError, match="generation failed"):
        _rgb(processor, 10)
    assert torch.equal(processor._output, _controls(0))
    model.fail = False
    with pytest.raises(RuntimeError, match="failed"):
        _rgb(processor, 10)
    assert len(model.steps) == 1


def test_a_model_without_a_simulation_is_owned_and_closed_with_the_processor(no_sim):
    processor, model = _processor()

    processor.close()
    processor.close()

    assert model.streams[0].closed == 1
    assert model.closed == 1


def test_a_simulation_model_is_shared_and_left_open(monkeypatch):
    model = RecordingModel(None)
    requested = []
    sim = SimpleNamespace(get_or_create_backend=lambda cfg: requested.append(cfg) or model)
    monkeypatch.setattr(processor_module, "SimulationContext", SimpleNamespace(instance=lambda: sim))
    cfg = ImageTransferProcessorCfg(backend=RecordingModelCfg())

    first = cfg.func(cfg, _buffers(_controls(0)))
    second = cfg.func(cfg, _buffers(_controls(0)))
    first.close()

    assert requested == [cfg.backend, cfg.backend]
    assert second._stream is not first._stream
    assert model.streams[0].closed == 1 and model.streams[1].closed == 0
    assert model.closed == 0


def test_depth_controls_feed_image_transfer(no_sim):
    """Near depth is white, far and missing depth black, and the transfer generates from those controls."""
    depth = torch.tensor([1.0, 9.0, float("inf"), 0.0]).view(1, 1, 4, 1).expand(2, 2, 4, 1).contiguous()
    buffers = {"distance_to_image_plane": ProxyArray(wp.from_torch(depth))}
    cfgs = [
        FunctionProcessorCfg(
            function=depth_to_control,
            params={"near": 1.0, "far": 9.0},
            inputs=["distance_to_image_plane"],
            outputs=["control"],
        ),
        ImageTransferProcessorCfg(backend=RecordingModelCfg()),
    ]

    # Construct and run in order, as a camera does: each processor reads the outputs of the earlier ones.
    for cfg in cfgs:
        buffers.update(cfg.func(cfg, buffers)(buffers))

    output = buffers["rgb"].torch
    assert output.shape == (2, 2, 4, 3)
    assert output[0, 0, :, 0].tolist() == [0, 255, 255, 255]
    with pytest.raises(ValueError, match="near < far"):
        depth_to_control(depth, near=2.0, far=1.0)


def test_a_view_resetting_mid_chunk_joins_the_next_chunk_and_queues_stay_bounded(no_sim):
    """With four-frame updates, a view that resets mid-chunk starts with its newest control at the next chunk."""
    processor, model = _processor(initial_frames=1, update_frames=4, max_pending_frames=8)
    _rgb(processor, 1)
    _rgb(processor, 2)
    processor.reset([1])
    for value in (3, 4, 5):
        _rgb(processor, value)

    chunks, resets, _ = model.steps[-1]
    assert [chunk.shape[0] for chunk in chunks] == [4, 1] and resets == (1,)
    assert int(chunks[1][0, 0, 0, 0]) == 5
    assert [len(queue) for queue in processor._pending] == [0, 0]
    # Both views now follow one four-frame cadence for as long as the episode runs.
    for value in range(6, 30):
        _rgb(processor, value)
    assert all([chunk.shape[0] for chunk in step[0]] == [4, 4] for step in model.steps[2:])
    assert max(len(queue) for queue in processor._pending) < 4
