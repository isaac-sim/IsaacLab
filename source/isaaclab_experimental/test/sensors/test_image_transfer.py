# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Image transfer scheduling, resets, and ownership through the modifier interface."""

from types import SimpleNamespace

import pytest
import torch
from isaaclab_experimental.image_transfer import ImageTransferModifierCfg, depth_to_control, srgb_to_linear
from isaaclab_experimental.image_transfer import modifier as modifier_module

from isaaclab.sim import BackendCfg
from isaaclab.utils import configclass
from isaaclab.utils.modifiers import ModifierCfg, ModifierChain

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
    monkeypatch.setattr(modifier_module, "SimulationContext", SimpleNamespace(instance=lambda: None))


def _modifier(**kwargs):
    cfg = ImageTransferModifierCfg(backend=RecordingModelCfg(), seed=7, **kwargs)
    modifier = cfg.func(cfg=cfg, data_dim=(2, 2, 3, 3), device="cpu")
    return modifier, modifier._owned_model


def _controls(value):
    return torch.full((2, 2, 3, 3), value, dtype=torch.uint8)


def test_chunks_follow_the_configured_cadence_and_hold_between_chunks(no_sim):
    """The first chunk uses initial_frames, later chunks update_frames; queued controls are owned copies."""
    modifier, model = _modifier(initial_frames=1, update_frames=2)
    controls = _controls(10)

    output = modifier(controls)
    controls.fill_(20)
    assert modifier(controls) is output
    assert len(model.steps) == 1
    assert torch.equal(output, _controls(245))
    controls.fill_(30)
    modifier(controls)

    assert model.opened == [(2, (7, 8))]
    assert [tuple(chunk.shape) for chunk in model.steps[1][0]] == [(2, 2, 3, 3)] * 2
    torch.testing.assert_close(model.steps[1][0][0][:, 0, 0, 0], torch.tensor([20, 30], dtype=torch.uint8))
    assert torch.equal(output, _controls(225))


def test_reset_restarts_only_the_selected_view_with_a_new_seed(no_sim):
    modifier, model = _modifier(initial_frames=1, update_frames=1)
    modifier(_controls(10))

    modifier.reset(torch.tensor([1]))
    assert torch.equal(modifier._output[1], torch.zeros(2, 3, 3, dtype=torch.uint8))
    assert torch.equal(modifier._output[0], torch.full((2, 3, 3), 245, dtype=torch.uint8))
    modifier(_controls(10))

    # View 1 starts again from its seed 8, advanced by the number of views for its second episode.
    assert model.steps[-1][1:] == ((1,), (10,))


def test_a_failed_generation_publishes_nothing_and_is_not_retried(no_sim):
    modifier, model = _modifier()
    model.fail = True

    with pytest.raises(RuntimeError, match="generation failed"):
        modifier(_controls(10))
    assert torch.equal(modifier._output, _controls(0))
    model.fail = False
    with pytest.raises(RuntimeError, match="failed"):
        modifier(_controls(10))
    assert len(model.steps) == 1


def test_a_model_without_a_simulation_is_owned_and_closed_with_the_modifier(no_sim):
    modifier, model = _modifier()

    modifier.close()
    modifier.close()

    assert model.streams[0].closed == 1
    assert model.closed == 1


def test_a_simulation_model_is_shared_and_left_open(monkeypatch):
    model = RecordingModel(None)
    requested = []
    sim = SimpleNamespace(get_or_create_backend=lambda cfg: requested.append(cfg) or model)
    monkeypatch.setattr(modifier_module, "SimulationContext", SimpleNamespace(instance=lambda: sim))
    cfg = ImageTransferModifierCfg(backend=RecordingModelCfg())

    first = cfg.func(cfg=cfg, data_dim=(2, 2, 3, 3), device="cpu")
    second = cfg.func(cfg=cfg, data_dim=(2, 2, 3, 3), device="cpu")
    first.close()

    assert requested == [cfg.backend, cfg.backend]
    assert second._stream is not first._stream
    assert model.streams[0].closed == 1 and model.streams[1].closed == 0
    assert model.closed == 0


def test_depth_controls_feed_image_transfer_in_a_chain(no_sim):
    """Near depth is white, far and missing depth black, and the chain generates from those controls."""
    depth = torch.tensor([1.0, 9.0, float("inf"), 0.0]).view(1, 1, 4, 1).expand(2, 2, 4, 1)
    chain = ModifierChain(
        [
            ModifierCfg(func=depth_to_control, params={"near": 1.0, "far": 9.0}),
            ImageTransferModifierCfg(backend=RecordingModelCfg()),
        ],
        "cpu",
    )

    output = chain(depth)

    assert output.shape == (2, 2, 4, 3)
    assert output[0, 0, :, 0].tolist() == [0, 255, 255, 255]
    with pytest.raises(ValueError, match="near < far"):
        depth_to_control(depth, near=2.0, far=1.0)


def test_srgb_to_linear_inverts_the_srgb_transfer_function():
    srgb = torch.tensor([0, 10, 128, 255], dtype=torch.uint8).view(1, 1, 4, 1).expand(1, 1, 4, 3)

    linear = srgb_to_linear(srgb)

    torch.testing.assert_close(linear[0, 0, :, 0], torch.tensor([0.0, 0.0030353, 0.2158605, 1.0]))
