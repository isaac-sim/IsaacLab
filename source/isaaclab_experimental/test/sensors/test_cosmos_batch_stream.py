# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched Cosmos camera views in the server: per-view frames, resets, and causal VAE caches.

A stand-in Framework model replaces Cosmos: its VAE keeps one cache per tokenizer like the Framework's, and its
generator echoes each view's control so views can be told apart.
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from isaaclab_experimental.cosmos.server import _framework

pytestmark = [
    pytest.mark.unit,
    # The server pins its work to the model's CUDA device; the stand-in model itself runs on the CPU.
    pytest.mark.skipif(not torch.cuda.is_available(), reason="The Cosmos server selects a CUDA device"),
]

HEIGHT, WIDTH = 480, 832


@dataclass
class StreamingTransferStep:
    control: torch.Tensor
    reset_rows: tuple[int, ...] = ()
    seeds: tuple[int, ...] | None = None


class CausalVae:
    """Holds the streaming caches like the Framework's video VAE: one per tokenizer, not per view."""

    def __init__(self):
        self._enc_cache = self._new_enc_cache()
        self._enc_stream_shape = None
        self._dec_cache = self._new_dec_cache()

    def _new_enc_cache(self):
        return [None]

    def _new_dec_cache(self):
        return [None]


class Tokenizer:
    def __init__(self):
        self.model = SimpleNamespace(model=CausalVae())

    @contextmanager
    def use_cached_encoder(self):
        yield

    @contextmanager
    def use_cached_decoder(self):
        yield


class Model:
    """Encodes a view's mean control value, generates it back, and decodes one or four frames per causal cache."""

    tensor_kwargs = {"dtype": torch.float32}

    def __init__(self):
        self.tokenizer_vision_gen = Tokenizer()
        self.vae = self.tokenizer_vision_gen.model.model
        self.encoded_caches: list[int] = []
        self.steps: list[tuple[tuple[int, ...], tuple[int, ...] | None]] = []
        self.iterators = 0

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        frames = pixels.shape[2]
        assert frames == (1 if self.vae._enc_stream_shape is None else 4), "a view's encoder cache was mixed up"
        self.vae._enc_stream_shape = "primed"
        self.encoded_caches.append(id(self.vae._enc_cache))
        return pixels.mean(dim=(1, 3, 4), keepdim=True)[:, :, -1:]  # [1,1,1,1,1]

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        frames = 1 if self.vae._dec_cache[0] is None else 4
        self.vae._dec_cache[0] = "primed"
        return latent.expand(1, 3, frames, HEIGHT, WIDTH)

    def iter_samples_from_batch_autoregressive_streaming_transfer(self, *, control_latent_chunks, seeds, **_):
        self.iterators += 1
        for step in control_latent_chunks:
            if isinstance(step, StreamingTransferStep):
                self.steps.append((step.reset_rows, step.seeds))
                step = step.control
            else:
                self.steps.append(((), None))
            yield {"vision": step.clone()}


@pytest.fixture
def batch(monkeypatch):
    """Two views on the stand-in model; ``partial`` selects the compiled (per-view reset) runtime."""
    module = types.ModuleType("cosmos_framework.model.generator.omni_mot_causal_model")
    module.StreamingTransferStep = StreamingTransferStep
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def make(views=2, partial=True, budget=13):
        model = Model()
        owner = SimpleNamespace(
            _pipeline=SimpleNamespace(model=model),
            _device=torch.device("cuda", torch.cuda.current_device()),
            capabilities={"partial_resets": partial},
            _stream=None,
        )
        stream = _framework._CosmosBatchStream(owner, {}, tuple(range(views)), HEIGHT, WIDTH, budget)
        return stream, model

    return make


def _controls(*frames_and_values):
    return [np.full((frames, HEIGHT, WIDTH, 3), value, dtype=np.uint8) for frames, value in frames_and_values]


def test_views_advance_together_and_a_restarting_view_sends_one_frame(batch):
    """View 1 restarts with one frame while view 0 continues with four; each keeps its own images."""
    stream, model = batch()

    first = stream.step(_controls((1, 40), (1, 200)), (), ())
    update = stream.step(_controls((4, 40), (4, 200)), (), ())
    restart = stream.step(_controls((4, 40), (1, 200)), (1,), (21,))

    assert [image.shape[0] for image in first] == [1, 1]
    assert [image.shape[0] for image in update] == [4, 4]
    assert [image.shape[0] for image in restart] == [4, 1]
    # Each view gets back its own control value, so batched views never mix.
    assert all(int(images[0].mean()) == 40 and int(images[1].mean()) == 200 for images in (first, update, restart))
    assert model.steps == [((), None), ((), None), ((1,), (21,))]
    assert stream._frames == [9, 1]


def test_each_view_keeps_its_own_causal_vae_cache_and_a_restart_starts_a_new_one(batch):
    stream, model = batch()
    stream.step(_controls((1, 1), (1, 2)), (), ())
    stream.step(_controls((4, 1), (4, 2)), (), ())
    view0, view1 = model.encoded_caches[0], model.encoded_caches[1]
    assert view0 != view1 and model.encoded_caches[2:4] == [view0, view1]

    stream.step(_controls((4, 1), (1, 2)), (1,), (5,))
    assert model.encoded_caches[4] == view0 and model.encoded_caches[5] not in (view0, view1)


def test_views_reject_mismatched_frames_early_resets_and_their_own_cap(batch):
    stream, _ = batch(budget=5)
    with pytest.raises(ValueError, match="start their first episode fresh"):
        stream.step(_controls((1, 1), (1, 1)), (1,), (3,))
    stream, _ = batch(budget=5)
    stream.step(_controls((1, 1), (1, 1)), (), ())
    with pytest.raises(ValueError, match="view 1 must be uint8 THWC with 4 frames"):
        stream.step(_controls((4, 1), (1, 1)), (), ())

    stream, _ = batch(budget=5)
    stream.step(_controls((1, 1), (1, 1)), (), ())
    stream.step(_controls((4, 1), (1, 1)), (1,), (3,))
    with pytest.raises(RuntimeError, match="horizon exceeded for view 0"):
        stream.step(_controls((4, 1), (4, 1)), (), ())


def test_eager_runtimes_restart_all_views_together_but_not_one(batch):
    """Without the compiled runtime a single view cannot restart; restarting every view starts a new batch."""
    stream, model = batch(partial=False)
    stream.step(_controls((1, 1), (1, 2)), (), ())
    stream.step(_controls((4, 1), (4, 2)), (), ())
    with pytest.raises(ValueError, match="without --no-compile"):
        stream.step(_controls((4, 1), (1, 2)), (1,), (7,))

    stream, model = batch(partial=False)
    stream.step(_controls((1, 1), (1, 2)), (), ())
    images = stream.step(_controls((1, 1), (1, 2)), (0, 1), (7, 8))
    assert [image.shape[0] for image in images] == [1, 1]
    assert model.iterators == 2 and model.steps[-1] == ((), None)
    assert stream._seeds == (7, 8)


def test_prompt_rows_merge_into_one_batch_with_a_caption_per_view():
    rows = [
        {"caption": ["A lab."], "neg_caption": [""], "system_prompt": "S", "fps": [30.0]},
        {"caption": ["A kitchen."], "neg_caption": [""], "system_prompt": "S", "fps": [30.0]},
    ]
    data = _framework._merge_prompt_rows(rows, "caption")
    assert data["caption"] == ["A lab.", "A kitchen."]
    assert data["neg_caption"] == ["", ""] and data["fps"] == [30.0, 30.0] and data["system_prompt"] == "S"
