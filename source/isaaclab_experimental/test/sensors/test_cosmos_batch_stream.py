# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched Cosmos camera views in the server: per-view frames, resets, and causal VAE caches.

A stand-in Framework model replaces Cosmos: its VAE keeps one cache per tokenizer like the Framework's, and its
generator echoes each view's control so views can be told apart.
"""

from __future__ import annotations

import gc
import sys
import types
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from isaaclab_experimental.cosmos.server import _framework

pytestmark = pytest.mark.unit

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
        try:
            yield
        finally:
            self.model.model._enc_cache = self.model.model._new_enc_cache()
            self.model.model._enc_stream_shape = None

    @contextmanager
    def use_cached_decoder(self):
        try:
            yield
        finally:
            self.model.model._dec_cache = self.model.model._new_dec_cache()


class Model:
    """Encodes a view's mean control value, generates it back, and decodes one or four frames per causal cache."""

    tensor_kwargs = {"dtype": torch.float32}
    input_caption_key = "caption"

    def __init__(self):
        self.tokenizer_vision_gen = Tokenizer()
        self.vae = self.tokenizer_vision_gen.model.model
        self.encoded_caches: list[int] = []
        self.cache_tensors: list[weakref.ReferenceType] = []
        self.steps: list[tuple[tuple[int, ...], tuple[int, ...] | None]] = []
        self.iterators = 0

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        frames = pixels.shape[2]
        assert frames == (1 if self.vae._enc_stream_shape is None else 4), "a view's encoder cache was mixed up"
        if self.vae._enc_cache[0] is None:
            self.vae._enc_cache[0] = torch.zeros(1)
            self.cache_tensors.append(weakref.ref(self.vae._enc_cache[0]))
        self.vae._enc_stream_shape = "primed"
        self.encoded_caches.append(id(self.vae._enc_cache))
        return pixels.mean(dim=(1, 3, 4), keepdim=True)[:, :, -1:]  # [1,1,1,1,1]

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        frames = 1 if self.vae._dec_cache[0] is None else 4
        if self.vae._dec_cache[0] is None:
            self.vae._dec_cache[0] = torch.zeros(1)
            self.cache_tensors.append(weakref.ref(self.vae._dec_cache[0]))
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
    # The server pins its work to the model's CUDA device; the stand-in model itself runs on the CPU.
    if not torch.cuda.is_available():
        pytest.skip("The Cosmos server selects a CUDA device")
    module = types.ModuleType("cosmos_framework.model.generator.omni_mot_causal_model")
    module.StreamingTransferStep = StreamingTransferStep
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def make(views=2, partial=True, budget=13, modality="depth"):
        model = Model()
        # Keep the real session admission and lifecycle; replace only the loaded Framework and text conditioning.
        owner = _framework.CosmosInferenceModel.__new__(_framework.CosmosInferenceModel)
        owner._pipeline = SimpleNamespace(model=model)
        owner._device = torch.device("cuda", torch.cuda.current_device())
        owner.capabilities = {"partial_resets": partial, "modalities": ["depth", "blur"]}
        owner._stream = None
        owner._closed = False
        monkeypatch.setattr(owner, "_prompt_batch", lambda text, *args: {"caption": [text]})
        stream = owner.open_stream(
            num_views=views,
            seeds=tuple(range(views)),
            prompt="A lab.",
            modality=modality,
            height=HEIGHT,
            width=WIDTH,
            max_episode_frames=budget,
        )
        return stream, model

    return make


def _controls(*frames_and_values):
    return [np.full((frames, HEIGHT, WIDTH, 3), value, dtype=np.uint8) for frames, value in frames_and_values]


def test_new_sessions_change_view_count_without_reloading_the_model(batch):
    """A resident model serves four views, then eight, then one, with fresh history and distinct images."""
    stream, model = batch(views=4)
    owner = stream._owner
    for views in (4, 8, 1):
        if views != 4:
            stream = owner.open_stream(
                num_views=views,
                seeds=tuple(range(views)),
                prompt="A lab.",
                modality="depth",
                height=HEIGHT,
                width=WIDTH,
                max_episode_frames=13,
            )
        try:
            for frames in (1, 4):
                images = stream.step(_controls(*((frames, 20 + view) for view in range(views))), (), ())
                assert len(images) == views
                for view, pixels in enumerate(images):
                    assert pixels.shape == (frames, HEIGHT, WIDTH, 3)
                    assert pixels.dtype == np.uint8
                    assert np.all(pixels == 20 + view)
        finally:
            stream.close()
    assert model.iterators == 3


def test_a_session_generates_its_whole_requested_budget_then_stops(batch):
    """The server sets no length limit: a 10-second episode at 60 captures/s gets all 601 frames."""
    stream, _ = batch(views=1, budget=601)
    try:
        stream.step(_controls((1, 40)), (), ())
        for _ in range(150):
            images = stream.step(_controls((4, 40)), (), ())
            assert images[0].shape == (4, HEIGHT, WIDTH, 3)
            assert np.all(images[0] == 40)
        with pytest.raises(RuntimeError, match="episode horizon exceeded"):
            stream.step(_controls((4, 40)), (), ())
    finally:
        stream.close()


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


def test_closing_a_batch_releases_every_views_caches(batch):
    stream, model = batch()
    stream.step(_controls((1, 40), (1, 200)), (), ())
    assert len(model.cache_tensors) == 4 and all(ref() is not None for ref in model.cache_tensors)
    stream.close()
    stream.close()
    gc.collect()
    # Keep the closed stream and resident model alive; neither may retain a view's cached tensors.
    assert all(ref() is None for ref in model.cache_tensors)


@pytest.mark.parametrize("window,frames", [(30, 121), (8, 33)])
def test_warmup_fills_the_history_window(window, frames, monkeypatch):
    """Warmup runs one latent past the history window, so the first real episode needs no new shapes."""
    model = _framework.CosmosInferenceModel.__new__(_framework.CosmosInferenceModel)
    model.capabilities = {"kv_window": window}
    opened = {}

    class Stream:
        def step(self, controls, reset_rows, seeds):
            opened["frames"] += controls[0].shape[0]

        def close(self):
            opened["closed"] = True

    def open_stream(**settings):
        opened.update(budget=settings["max_episode_frames"], frames=0)
        return Stream()

    monkeypatch.setattr(model, "open_stream", open_stream)
    model.warmup(height=4, width=4)
    assert opened == {"budget": frames, "frames": frames, "closed": True}


def test_prompt_rows_merge_into_one_batch_with_a_caption_per_view():
    rows = [
        {"caption": ["A lab."], "neg_caption": [""], "system_prompt": "S", "fps": [30.0]},
        {"caption": ["A kitchen."], "neg_caption": [""], "system_prompt": "S", "fps": [30.0]},
    ]
    data = _framework._merge_prompt_rows(rows, "caption")
    assert data["caption"] == ["A lab.", "A kitchen."]
    assert data["neg_caption"] == ["", ""] and data["fps"] == [30.0, 30.0] and data["system_prompt"] == "S"


def test_blur_views_are_blurred_by_the_framework_filter_before_encoding(batch, monkeypatch):
    """The service turns each view's RGB into the blur control with the Framework's own augmentor (medium preset)."""
    calls = []

    def augment(frames, *, hint_key, preset_edge_threshold, preset_blur_strength):
        calls.append((tuple(frames.shape), hint_key, preset_blur_strength))
        return 255 - frames  # stands in for the blur; [3,T,H,W] uint8

    args = types.ModuleType("cosmos_framework.inference.args")
    args.TransferHintKey = SimpleNamespace(BLUR="blur")
    args.PresetBlurStrength = SimpleNamespace(MEDIUM="medium")
    args.PresetEdgeThreshold = SimpleNamespace(MEDIUM="medium")
    transfer = types.ModuleType("cosmos_framework.inference.transfer")
    transfer.apply_transfer_control_augmentor = augment
    monkeypatch.setitem(sys.modules, args.__name__, args)
    monkeypatch.setitem(sys.modules, transfer.__name__, transfer)

    stream, _ = batch(modality="blur")
    images = stream.step(_controls((1, 40), (1, 200)), (), ())

    assert calls == [((3, 1, HEIGHT, WIDTH), "blur", "medium")] * 2
    # The stand-in model echoes what it encoded: the blurred (here inverted) RGB of each view.
    assert int(images[0].mean()) == 215 and int(images[1].mean()) == 55
