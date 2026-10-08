# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resident Sim-Transfer inference, imported only in the Cosmos model environment."""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Generator, Iterator
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any

import numpy as np

from .._protocol import CANVAS_ASPECT_RATIOS as _CANVASES
from .._protocol import DEFAULT_MAX_EPISODE_FRAMES

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)



class CosmosInferenceModel:
    """Load a causal distilled Cosmos checkpoint once and serve one camera session.

    Construct this resource in the Cosmos Framework environment, independently of the
    simulation process. Each stream owns its VAE and transformer history; closing or
    resetting a stream preserves the resident weights. Concurrent streams, multiple
    views, and partial episode resets are unsupported.
    """

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda:0",
        *,
        use_compile: bool = True,
        max_episode_frames: int | None = DEFAULT_MAX_EPISODE_FRAMES,
    ):
        """Load weights on the selected CUDA device.

        Args:
            checkpoint: Exported HF checkpoint directory or a Framework checkpoint name.
            device: CUDA device in this process's visible GPU set.
            use_compile: Enable the Framework's compiled CUDA-graph streaming path.
            max_episode_frames: Longest episode a session may request, ``1 + 4*k`` frames. None removes the
                cap. The model was trained on 201-frame episodes; longer episodes are unvalidated.

        Raises:
            ValueError: If the runtime or checkpoint does not support this streaming contract.
            ImportError: If the optional Cosmos Framework runtime is unavailable.
        """
        if max_episode_frames is not None and (
            type(max_episode_frames) is not int or max_episode_frames < 1 or (max_episode_frames - 1) % 4
        ):
            raise ValueError("The Cosmos episode cap must be 1 + 4*k frames, or None for no cap.")
        self._max_episode_frames = max_episode_frames
        if max_episode_frames is None or max_episode_frames > DEFAULT_MAX_EPISODE_FRAMES:
            logger.warning(
                "Cosmos episodes may exceed the model's trained horizon of %d frames; check quality and memory use.",
                DEFAULT_MAX_EPISODE_FRAMES,
            )
        import torch
        from cosmos_framework.inference.common.init import init_script

        selected_device = torch.device(device)
        if selected_device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("Cosmos Sim-Transfer requires an available CUDA device.")
        if selected_device.index is None:
            selected_device = torch.device("cuda", torch.cuda.current_device())
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("The Cosmos service requires a single unsharded worker, outside torchrun.")
        torch.cuda.set_device(selected_device)
        init_script()

        from cosmos_framework.inference.args import OmniSetupOverrides

        self._closed = False
        self._device = selected_device
        self._stream: _CosmosInferenceStream | None = None
        self._pipeline = None
        self._metadata_dir = tempfile.TemporaryDirectory(prefix="isaaclab-cosmos-")
        try:
            setup = OmniSetupOverrides(
                checkpoint_path=checkpoint,
                output_dir=self._metadata_dir.name,
                use_torch_compile=use_compile,
                use_cuda_graphs=use_compile,
                compiled_region="language",
                experiment_overrides=[
                    "model.config.kv_cache_inference_size=8",
                    "model.config.attention_sink_size=3",
                ],
                dp_shard_size=1,
                cp_size=1,
                cfgp_size=1,
                guardrails=False,
                diffusion_cache=False,
            ).build_setup()
            logger.info("Loading Cosmos checkpoint %s on %s", checkpoint, selected_device)
            self._pipeline = setup.get_inference_cls().create(setup)
            model = self._pipeline.model
            config = model.config
            if not (
                config.video_temporal_causal
                and config.fixed_step_sampler_config is not None
                and config.teacher_forcing_frames_per_chunk == 1
                and config.tokenizer.temporal_compression_factor == 4
                and config.kv_cache_inference_size == 8
                and config.attention_sink_size == 3
                and config.kv_cache_dtype is None
            ):
                raise ValueError(
                    "Use a distilled causal Sim-Transfer checkpoint with one latent per chunk, "
                    "4x temporal compression, and a BF16 finite history window."
                )
            if use_compile and not (
                config.compile.enabled
                and config.compile.use_cuda_graphs
                and config.compile.ar_post_saturation_mode == "default"
            ):
                raise ValueError("Compiled Sim-Transfer requires the default CUDA-graph AR runtime.")
            for owner, method in (
                (model, "iter_samples_from_batch_autoregressive_streaming_transfer"),
                (model, "_get_teacher_forcing_replay_policy"),
                (model.tokenizer_vision_gen, "use_cached_encoder"),
                (model.tokenizer_vision_gen, "use_cached_decoder"),
            ):
                if not callable(getattr(owner, method, None)):
                    raise ValueError(f"The Cosmos checkpoint runtime does not provide {method}().")
            policy = model._get_teacher_forcing_replay_policy()
            if not (
                policy.control_visibility == "causal"
                and policy.controls_read_strict_past_clean_rgb
                and policy.clean_pass_causality == "frame"
            ):
                raise ValueError("Cosmos requires causal framewise controls with strict-past generated RGB history.")
            model.eval()
            torch.cuda.synchronize(selected_device)
        except BaseException:
            self.close()
            raise
        self.capabilities: dict[str, Any] = {
            "num_views": 1,
            "max_concurrent_streams": 1,
            "modalities": ["edge", "depth", "seg"],
            "initial_frames": 1,
            "update_frames": 4,
            "canvases": [list(canvas) for canvas in _CANVASES],
            "max_episode_frames": max_episode_frames,
            "fps": 30,
            "partial_resets": False,
        }

    def open_stream(
        self,
        *,
        num_views: int,
        seeds: tuple[int, ...],
        prompt: str | None,
        modality: str,
        height: int,
        width: int,
        max_episode_frames: int,
    ) -> _CosmosInferenceStream:
        """Open one fixed-resolution generation session with a fresh episode history.

        Controls must already represent the selected modality. The Framework's
        transfer system prompt and native duration/resolution formatting condition
        generation; no reference video or cookbook files are loaded.

        Args:
            num_views: Camera count; must be one.
            seeds: One seed for the initial episode.
            prompt: Appearance description. None uses empty text plus native metadata.
            modality: Preprocessed control type: edge, depth, or seg.
            height: Control image height [pixels].
            width: Control image width [pixels].
            max_episode_frames: Episode horizon; must be 1 + 4*k and within the service's cap.

        Returns:
            A session whose close releases generation history, retaining model weights.
        """
        if self._closed:
            raise RuntimeError("The Cosmos model is closed.")
        if self._stream is not None:
            raise RuntimeError("The Cosmos model already has an active camera session.")
        if type(num_views) is not int or num_views != 1 or len(seeds) != 1:
            raise ValueError("Cosmos currently supports exactly one camera view and seed per service.")
        _validate_seed(seeds[0])
        if prompt is not None and not isinstance(prompt, str):
            raise ValueError("The Cosmos appearance prompt must be text or None.")
        if modality not in self.capabilities["modalities"]:
            raise ValueError("Cosmos controls must use edge, depth, or seg modality.")
        if (height, width) not in _CANVASES:
            raise ValueError(f"Unsupported Cosmos canvas {(height, width)}; choose one of {list(_CANVASES)}.")
        if (
            type(max_episode_frames) is not int
            or max_episode_frames < 1
            or (max_episode_frames - 1) % 4
            or (self._max_episode_frames is not None and max_episode_frames > self._max_episode_frames)
        ):
            raise ValueError(
                f"Cosmos max_episode_frames must be 1 + 4*k and at most the service cap {self._max_episode_frames}; "
                "raise it with isaaclab-cosmos-server --max-episode-frames."
            )

        import torch

        with torch.cuda.device(self._device):
            data = self._prompt_batch(prompt, modality, height, width, max_episode_frames)
        self._stream = _CosmosInferenceStream(self, data, seeds[0], height, width, max_episode_frames, modality)
        return self._stream

    def _prompt_batch(
        self, prompt: str | None, modality: str, height: int, width: int, max_episode_frames: int
    ) -> dict[str, Any]:
        """Build one episode's text conditioning: appearance prompt, control instruction, and metadata."""
        from cosmos_framework.inference.args import OmniSampleOverrides
        from cosmos_framework.inference.inference import _get_prompt_sample_data
        from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import _SYSTEM_PROMPT_TRANSFER

        model = self._pipeline.model
        # Build the native prompt metadata without a transfer file: controls arrive
        # incrementally over the service boundary rather than through media loading.
        sample = OmniSampleOverrides(
            name="isaaclab-camera",
            output_dir=self._metadata_dir.name,
            model_mode="video2video",
            prompt=prompt or "",
            resolution="480",
            aspect_ratio=_CANVASES[(height, width)],
            num_frames=max_episode_frames,
            fps=30,
            num_steps=4,
            guidance=1.0,
            shift=5.0,
            negative_metadata_mode="none",
            negative_prompt_keep_metadata=False,
            prompt_upsampling=False,
            **({"autoregressive": False} if "autoregressive" in OmniSampleOverrides.model_fields else {}),
        ).build_sample(model_config=model.config)
        data = _get_prompt_sample_data(sample, model, h=height, w=width, device="cuda")
        # Native streaming transfer names the active hint in the caption; its
        # transformer batch carries no separate modality identifier.
        data[model.input_caption_key][0] = (
            data[model.input_caption_key][0].rstrip()
            + f" Follow the {modality} control video precisely: shape, contour, silhouette, position, and "
            f"motion of every visible structure must align with the {modality} signal at every frame."
        )
        data.update(dataset_name="video_transfer", system_prompt=_SYSTEM_PROMPT_TRANSFER, fps=[30.0])
        return data

    def warmup(self, *, height: int = 480, width: int = 832) -> None:
        """Warm one canvas through finite-history saturation using a disposable session.

        The warmup emits up to 33 frames from blank controls, within the episode cap. Its
        prompt, seed, VAE caches, and generation history are discarded before real cameras
        can connect. Other canvases or prompts can still require compilation on their first use.
        """
        frames = 33 if self._max_episode_frames is None else min(33, self._max_episode_frames)
        stream = self.open_stream(
            num_views=1,
            seeds=(0,),
            prompt=None,
            modality="edge",
            height=height,
            width=width,
            max_episode_frames=frames,
        )
        try:
            stream.step([np.zeros((1, height, width, 3), dtype=np.uint8)], (), ())
            controls = np.zeros((4, height, width, 3), dtype=np.uint8)
            for _ in range((frames - 1) // 4):
                stream.step([controls], (), ())
        finally:
            stream.close()

    def close(self) -> None:
        """Close any session and release the resident model and owned metadata directory."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._stream is not None:
                self._stream.close()
        finally:
            self._pipeline = None
            self._metadata_dir.cleanup()


def _validate_seed(seed: int) -> None:
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Cosmos episode seeds must be integers in [0, 2**63).")


_KEEP_PROMPT = object()
"""Marker for a reset that keeps the current episode prompt."""


class _CosmosInferenceStream:
    """Own one single-view causal VAE session and autoregressive generation iterator."""

    def __init__(
        self,
        owner: CosmosInferenceModel,
        data_batch: dict[str, Any],
        seed: int,
        height: int,
        width: int,
        max_episode_frames: int,
        modality: str,
    ):
        self._owner = owner
        self._model = owner._pipeline.model
        self._data_batch = data_batch
        self._seed = seed
        self._canvas = (height, width, 3)
        self._max_episode_frames = max_episode_frames
        self._modality = modality
        self._frame_count = 0
        self._closed = False
        self._cache_scope: ExitStack | None = None
        self._next_control: torch.Tensor | None = None
        self._iterator = self._make_iterator()

    def step(
        self,
        controls: list[np.ndarray],
        reset_rows: tuple[int, ...],
        seeds: tuple[int, ...],
        *,
        prompt: str | None | object = _KEEP_PROMPT,
    ) -> list[np.ndarray]:
        """Generate uint8 THWC RGB from one initial frame or four subsequent frames.

        A full reset ``reset_rows=(0,)`` requires one new seed and a one-frame
        control. It closes the previous generation and VAE state before processing
        the new episode, and with ``prompt`` it also rebuilds the episode's text
        conditioning. Input arrays are never mutated. Model failures close this
        session because an inference step cannot be safely replayed.
        """
        if prompt is not _KEEP_PROMPT and (not reset_rows or (prompt is not None and not isinstance(prompt, str))):
            raise ValueError("A Cosmos prompt must be text or None and can change only with an episode reset.")
        if self._closed:
            raise RuntimeError("The Cosmos camera session is closed.")
        if reset_rows not in ((), (0,)) or len(seeds) != len(reset_rows):
            raise ValueError("Cosmos supports a full single-view reset with exactly one replacement seed.")
        if seeds:
            _validate_seed(seeds[0])
        expected_frames = 1 if self._frame_count == 0 or reset_rows else 4
        expected_shape = (expected_frames, *self._canvas)
        if len(controls) != 1 or not isinstance(controls[0], np.ndarray):
            raise ValueError("Cosmos requires exactly one numpy control array.")
        control = controls[0]
        if control.dtype != np.uint8 or control.shape != expected_shape:
            raise ValueError(f"Cosmos controls must be uint8 THWC with shape {expected_shape}.")
        previous_frames = 0 if reset_rows else self._frame_count
        if previous_frames + expected_frames > self._max_episode_frames:
            raise RuntimeError("Cosmos episode horizon exceeded; reset the camera before sending more controls.")

        import torch

        try:
            # Service connections run in separate threads, whose default CUDA
            # device can differ from the worker's selected model device.
            with torch.cuda.device(self._owner._device), torch.inference_mode():
                if reset_rows:
                    self._close_episode()
                    if prompt is not _KEEP_PROMPT:
                        height, width, _ = self._canvas
                        self._data_batch = self._owner._prompt_batch(
                            prompt, self._modality, height, width, self._max_episode_frames
                        )
                    self._seed = seeds[0]
                    self._frame_count = 0
                    self._iterator = self._make_iterator()
                if self._cache_scope is None:
                    self._cache_scope = ExitStack()
                    self._cache_scope.enter_context(self._model.tokenizer_vision_gen.use_cached_encoder())
                    self._cache_scope.enter_context(self._model.tokenizer_vision_gen.use_cached_decoder())
                pixels = torch.from_numpy(np.ascontiguousarray(control)).permute(3, 0, 1, 2).unsqueeze(0)
                pixels = pixels.to(**self._model.tensor_kwargs).div(127.5).sub(1)
                latent = self._model.encode(pixels)
                self._next_control = latent[:, :, -1:].clone()
                generated = next(self._iterator)["vision"].clone()
                decoded = self._model.decode(generated)
                if tuple(decoded.shape) != (1, 3, expected_frames, *self._canvas[:2]):
                    raise RuntimeError(f"Cosmos decoder returned unexpected shape {tuple(decoded.shape)}.")
                packed = decoded[0].clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)
                video = packed.permute(1, 2, 3, 0).contiguous().cpu().numpy()
                self._frame_count = previous_frames + expected_frames
                return [video]
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Discard this session's history and allow a new camera to use the loaded model."""
        if self._closed:
            return
        self._closed = True
        try:
            self._close_episode()
        finally:
            if self._owner._stream is self:
                self._owner._stream = None
            self._model = None

    def _make_iterator(self) -> Generator[dict[str, torch.Tensor], None, None]:
        latent_frames = (self._max_episode_frames - 1) // 4 + 1
        return self._model.iter_samples_from_batch_autoregressive_streaming_transfer(
            data_batch=self._data_batch,
            control_latent_chunks=self._control_iterator(),
            num_frames=latent_frames,
            seeds=[self._seed],
            guidance=1.0,
            num_steps=4,
            shift=5.0,
            normalize_cfg=False,
            sampler_mode="distilled",
        )

    def _close_episode(self) -> None:
        try:
            self._iterator.close()
        finally:
            if self._cache_scope is not None:
                self._cache_scope.close()
                self._cache_scope = None
            self._next_control = None

    def _control_iterator(self) -> Iterator[torch.Tensor]:
        while True:
            if self._next_control is None:
                raise RuntimeError("Cosmos requested controls outside an explicit camera step.")
            control, self._next_control = self._next_control, None
            yield control
