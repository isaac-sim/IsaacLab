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
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

_CANVASES = {
    (480, 832): "16,9",
    (544, 736): "4,3",
    (640, 640): "1,1",
    (736, 544): "3,4",
    (832, 480): "9,16",
}

_MAX_BATCH_LATENT_FRAMES = 1 << 30
"""Maximum latent frames in a batched session. Each view's episode ends at its own cap, so the batch need not stop."""


class CosmosInferenceModel:
    """Load a causal distilled Cosmos checkpoint once and serve one camera session.

    Construct this resource in the Cosmos Framework environment, independently of the
    simulation process. Each stream owns its VAE and transformer history; closing or
    resetting a stream preserves the resident weights. Each session requests its camera count,
    with memory allocated for that batch. Resetting some views while others continue needs the
    compiled runtime. Concurrent sessions are unsupported.
    """

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda:0",
        *,
        use_compile: bool = True,
        kv_window: int = 30,
        attention_sink: int = 3,
    ):
        """Load weights on the selected CUDA device.

        Args:
            checkpoint: Exported HF checkpoint directory or a Framework checkpoint name.
            device: CUDA device in this process's visible GPU set.
            use_compile: Enable the Framework's compiled CUDA-graph streaming path.
            kv_window: Generation history the transformer attends to [latent frames]. The Sim-Transfer recipe
                uses 30; shorter windows are faster and need less memory but remember less of the episode.
            attention_sink: Earliest latent frames always kept in the history window; the recipe uses 3.

        Raises:
            ValueError: If the runtime or checkpoint does not support this streaming contract.
            ImportError: If the optional Cosmos Framework runtime is unavailable.
        """
        if (
            type(kv_window) is not int
            or type(attention_sink) is not int
            or kv_window < 1
            or not 0 <= attention_sink < kv_window
        ):
            raise ValueError("Cosmos needs a positive history window and an attention sink smaller than it.")
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
                    f"model.config.kv_cache_inference_size={kv_window}",
                    f"model.config.attention_sink_size={attention_sink}",
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
                and config.kv_cache_inference_size == kv_window
                and config.attention_sink_size == attention_sink
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
            "max_concurrent_streams": 1,
            "modalities": ["edge", "blur", "depth", "seg"],
            "initial_frames": 1,
            "update_frames": 4,
            "canvases": [list(canvas) for canvas in _CANVASES],
            "fps": 30,
            # The Framework restarts single rows of a batch only on its compiled CUDA-graph path.
            "partial_resets": use_compile and _vae_cache_owner(model.tokenizer_vision_gen) is not None,
            "kv_window": kv_window,
            "attention_sink": attention_sink,
        }

    def open_stream(
        self,
        *,
        num_views: int,
        seeds: tuple[int, ...],
        prompt: str | None | list[str | None],
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
            num_views: Positive camera count. Views are generated as one batch, subject to available GPU memory.
                A new session may request a different count without reloading the model; its first step may compile.
            seeds: One seed per view for its first episode.
            prompt: Appearance description, the same for every view or one per view. None uses empty text plus
                native metadata. A view keeps its prompt across resets when the session has several views.
            modality: Control type: edge, depth, or seg prepared by the camera, or blur, which the service
                derives from the camera's RGB with the Framework's own filter.
            height: Control image height [pixels].
            width: Control image width [pixels].
            max_episode_frames: Episode horizon, ``1 + 4*k`` frames; the model is told the episode's duration.

        Returns:
            A session whose close releases generation history, retaining model weights.
        """
        if self._closed:
            raise RuntimeError("The Cosmos model is closed.")
        if self._stream is not None:
            raise RuntimeError("The Cosmos model already has an active camera session.")
        if type(num_views) is not int or num_views < 1 or len(seeds) != num_views:
            raise ValueError("Cosmos requires a positive camera count, with one seed per view.")
        if num_views > 1 and _vae_cache_owner(self._pipeline.model.tokenizer_vision_gen) is None:
            raise ValueError(
                "This Cosmos Framework's video tokenizer does not expose per-stream caches; serve one view."
            )
        for seed in seeds:
            _validate_seed(seed)
        prompts = list(prompt) if isinstance(prompt, (list, tuple)) else [prompt] * num_views
        if len(prompts) != num_views or any(text is not None and not isinstance(text, str) for text in prompts):
            raise ValueError("The Cosmos appearance prompt must be text or None, or one per camera view.")
        if modality not in self.capabilities["modalities"]:
            raise ValueError("Cosmos controls must use edge, blur, depth, or seg modality.")
        if (height, width) not in _CANVASES:
            raise ValueError(f"Unsupported Cosmos canvas {(height, width)}; choose one of {list(_CANVASES)}.")
        if type(max_episode_frames) is not int or max_episode_frames < 1 or (max_episode_frames - 1) % 4:
            raise ValueError("Cosmos max_episode_frames must be 1 + 4*k.")

        import torch

        with torch.cuda.device(self._device):
            rows = {text: self._prompt_batch(text, height, width, max_episode_frames) for text in set(prompts)}
        if num_views == 1:
            data = rows[prompts[0]]
            self._stream = _CosmosInferenceStream(self, data, seeds[0], height, width, max_episode_frames, modality)
        else:
            data = _merge_prompt_rows([rows[text] for text in prompts], self._pipeline.model.input_caption_key)
            self._stream = _CosmosBatchStream(self, data, seeds, height, width, max_episode_frames, modality)
        return self._stream

    def warmup(self, *, height: int = 480, width: int = 832) -> None:
        """Warm one canvas through finite-history saturation using a disposable session.

        The warmup emits ``1 + 4 * kv_window`` frames from blank controls (121 for the default window), one latent
        more than the history window holds. Its prompt, seed, VAE caches, and generation history are discarded
        before real cameras can connect. Other canvases or prompts can still require compilation on their first use.
        """
        frames = 1 + 4 * self.capabilities["kv_window"]
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

    def _prompt_batch(
        self, prompt: str | None, height: int, width: int, max_episode_frames: int
    ) -> dict[str, Any]:
        """Build one episode's text conditioning: appearance prompt and metadata."""
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
        # Like the Sim-Transfer recipe (emphasize_control_in_prompt off), the caption does not name the control.
        data.update(dataset_name="video_transfer", system_prompt=_SYSTEM_PROMPT_TRANSFER, fps=[30.0])
        return data


def _blur_controls(frames: np.ndarray) -> np.ndarray:
    """Turn uint8 THWC RGB into the blur control with the Framework's own filter, as its Transfer inference does.

    The recipe's medium preset has no random parameters, so filtering chunk by chunk matches filtering the video.
    """
    import torch
    from cosmos_framework.inference.args import PresetBlurStrength, PresetEdgeThreshold, TransferHintKey
    from cosmos_framework.inference.transfer import apply_transfer_control_augmentor

    blurred = apply_transfer_control_augmentor(
        torch.from_numpy(np.ascontiguousarray(frames.transpose(3, 0, 1, 2))),  # [3,T,H,W]
        hint_key=TransferHintKey.BLUR,
        preset_edge_threshold=PresetEdgeThreshold.MEDIUM,
        preset_blur_strength=PresetBlurStrength.MEDIUM,
    )
    return torch.as_tensor(blurred).to(torch.uint8).permute(1, 2, 3, 0).contiguous().numpy()  # [T,H,W,3]


def _merge_prompt_rows(rows: list[dict[str, Any]], caption_key: str) -> dict[str, Any]:
    """Combine one-view prompt batches into one batch with a caption, negative caption, and frame rate per view."""
    data = dict(rows[0])
    data[caption_key] = [row[caption_key][0] for row in rows]
    negative_key = f"neg_{caption_key}"
    if negative_key in data:
        data[negative_key] = [
            row[negative_key][0] if isinstance(row[negative_key], list) else row[negative_key] for row in rows
        ]
    data["fps"] = [30.0] * len(rows)
    return data


def _vae_cache_owner(tokenizer: Any) -> Any | None:
    """Return the video VAE object that holds the streaming encoder and decoder caches, or None.

    The Framework keeps one causal cache per tokenizer, sized for one batch; batched sessions swap a cache per view.
    """
    if getattr(tokenizer, "_decoder_override", None) is not None:
        return None
    owner = tokenizer
    for _ in range(4):
        if callable(getattr(owner, "_new_enc_cache", None)) and callable(getattr(owner, "_new_dec_cache", None)):
            return owner
        owner = getattr(owner, "model", None)
        if owner is None:
            return None
    return None


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
                            prompt, height, width, self._max_episode_frames
                        )
                    self._seed = seeds[0]
                    self._frame_count = 0
                    self._iterator = self._make_iterator()
                if self._cache_scope is None:
                    self._cache_scope = ExitStack()
                    self._cache_scope.enter_context(self._model.tokenizer_vision_gen.use_cached_encoder())
                    self._cache_scope.enter_context(self._model.tokenizer_vision_gen.use_cached_decoder())
                if self._modality == "blur":
                    control = _blur_controls(control)
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


class _CosmosBatchStream(_CosmosInferenceStream):
    """Own several camera views: one transformer batch with per-view episode restarts and a VAE state per view.

    All views advance together by one latent per step: a view starting or restarting its episode sends one frame
    while the others send four. The Framework restarts single rows of the transformer batch, but keeps one causal
    VAE cache for the whole batch, so each view's encoder and decoder caches are swapped in for its own calls.
    """

    def __init__(
        self,
        owner: CosmosInferenceModel,
        data_batch: dict[str, Any],
        seeds: tuple[int, ...],
        height: int,
        width: int,
        max_episode_frames: int,
        modality: str,
    ):
        self._owner = owner
        self._model = owner._pipeline.model
        self._vae = _vae_cache_owner(self._model.tokenizer_vision_gen)
        self._modality = modality
        self._data_batch = data_batch
        self._seeds = tuple(seeds)
        self._canvas = (height, width, 3)
        self._max_episode_frames = max_episode_frames
        self._frames = [0] * len(seeds)
        self._started = False
        self._closed = False
        self._cache_scope: ExitStack | None = None
        self._next_control = None
        self._vae_states = [self._fresh_vae_state() for _ in seeds]
        self._iterator = self._make_iterator()

    def step(
        self,
        controls: list[np.ndarray],
        reset_rows: tuple[int, ...],
        seeds: tuple[int, ...],
        **episode: object,
    ) -> list[np.ndarray]:
        """Generate uint8 THWC RGB for every view: one frame for starting views, four for the others.

        ``reset_rows`` restarts those views' episodes with ``seeds``; the other views continue. Restarting some
        views needs the compiled runtime; restarting all views also works eagerly. Views keep their prompts.
        """
        num_views = len(self._frames)
        if episode:
            raise ValueError("Cosmos views of a batched session keep their prompts across resets.")
        if self._closed:
            raise RuntimeError("The Cosmos camera session is closed.")
        if (
            not isinstance(reset_rows, tuple)
            or list(reset_rows) != sorted(set(reset_rows))
            or any(type(row) is not int or not 0 <= row < num_views for row in reset_rows)
            or len(seeds) != len(reset_rows)
        ):
            raise ValueError("Cosmos resets name distinct views in increasing order, with one seed each.")
        for seed in seeds:
            _validate_seed(seed)
        if reset_rows and not self._started:
            raise ValueError("Cosmos views start their first episode fresh; reset them after the first step.")
        restart_all = len(reset_rows) == num_views
        if reset_rows and not restart_all and not self._owner.capabilities["partial_resets"]:
            raise ValueError(
                "Resetting some camera views while others continue needs the compiled Cosmos server; start "
                "isaaclab-cosmos-server without --no-compile."
            )
        import torch

        resets = set(reset_rows)
        expected = [1 if not self._started or row in resets else 4 for row in range(num_views)]
        if len(controls) != num_views:
            raise ValueError(f"Cosmos requires one control chunk per view, {num_views} in total.")
        for row, control in enumerate(controls):
            if not isinstance(control, np.ndarray):
                raise ValueError("Cosmos controls must be NumPy arrays.")
            if control.dtype != np.uint8 or tuple(control.shape) != (expected[row], *self._canvas):
                raise ValueError(f"Cosmos controls of view {row} must be uint8 THWC with {expected[row]} frames.")
            if (0 if row in resets else self._frames[row]) + expected[row] > self._max_episode_frames:
                raise RuntimeError(f"Cosmos episode horizon exceeded for view {row}; reset it before continuing.")

        try:
            with torch.cuda.device(self._owner._device), torch.inference_mode():
                model_resets = tuple(reset_rows)
                if restart_all and not self._owner.capabilities["partial_resets"]:
                    # Eager runtimes cannot restart rows; restarting every view starts a new batch instead.
                    self._close_episode()
                    self._seeds = tuple(seeds)
                    self._iterator = self._make_iterator()
                    model_resets = ()
                if self._cache_scope is None:
                    self._cache_scope = ExitStack()
                    self._cache_scope.enter_context(self._model.tokenizer_vision_gen.use_cached_encoder())
                    self._cache_scope.enter_context(self._model.tokenizer_vision_gen.use_cached_decoder())
                for row in resets:
                    self._vae_states[row] = self._fresh_vae_state()
                latents = []
                for row, control in enumerate(controls):
                    if self._modality == "blur":
                        control = _blur_controls(control)
                    pixels = torch.from_numpy(np.ascontiguousarray(control))
                    pixels = pixels.permute(3, 0, 1, 2).unsqueeze(0).to(**self._model.tensor_kwargs).div(127.5).sub(1)
                    with self._view_vae(row):
                        latents.append(self._model.encode(pixels)[:, :, -1:])
                control_latent = torch.cat(latents, dim=0)  # [views,C,1,H,W]
                if model_resets:
                    from cosmos_framework.model.generator.omni_mot_causal_model import StreamingTransferStep

                    self._next_control = StreamingTransferStep(
                        control=control_latent, reset_rows=model_resets, seeds=tuple(seeds)
                    )
                else:
                    self._next_control = control_latent
                generated = next(self._iterator)["vision"].clone()  # [views,C,1,H,W]
                videos = []
                for row in range(num_views):
                    with self._view_vae(row):
                        decoded = self._model.decode(generated[row : row + 1])
                    if tuple(decoded.shape) != (1, 3, expected[row], *self._canvas[:2]):
                        raise RuntimeError(f"Cosmos decoder returned unexpected shape {tuple(decoded.shape)}.")
                    packed = decoded[0].clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)
                    videos.append(packed.permute(1, 2, 3, 0).contiguous().cpu().numpy())
                for row in range(num_views):
                    self._frames[row] = (0 if row in resets else self._frames[row]) + expected[row]
                self._started = True
                return videos
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Discard every view's history and causal VAE caches and allow a new camera to use the loaded model."""
        try:
            super().close()
        finally:
            self._vae_states = []
            self._vae = None

    def _make_iterator(self) -> Generator[dict[str, torch.Tensor], None, None]:
        return self._model.iter_samples_from_batch_autoregressive_streaming_transfer(
            data_batch=self._data_batch,
            control_latent_chunks=self._control_iterator(),
            num_frames=_MAX_BATCH_LATENT_FRAMES,
            seeds=list(self._seeds),
            guidance=1.0,
            num_steps=4,
            shift=5.0,
            normalize_cfg=False,
            sampler_mode="distilled",
        )

    def _fresh_vae_state(self) -> tuple[list, None, list]:
        return self._vae._new_enc_cache(), None, self._vae._new_dec_cache()

    @contextmanager
    def _view_vae(self, row: int) -> Iterator[None]:
        """Run the VAE with one view's causal encoder and decoder caches."""
        vae = self._vae
        vae._enc_cache, vae._enc_stream_shape, vae._dec_cache = self._vae_states[row]
        try:
            yield
        finally:
            self._vae_states[row] = (vae._enc_cache, vae._enc_stream_shape, vae._dec_cache)
