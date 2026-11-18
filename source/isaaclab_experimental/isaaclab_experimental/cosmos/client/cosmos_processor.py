# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cosmos image generation applied to camera images."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab_experimental.image_transfer import ImageTransferProcessor

from .._protocol import MAX_CHUNK_FRAMES

if TYPE_CHECKING:
    from isaaclab.utils.warp import ProxyArray

    from .cosmos_processor_cfg import CosmosTransferProcessorCfg

_SEED_MODULUS = 2**31


class CosmosTransferProcessor(ImageTransferProcessor):
    """Generate Cosmos camera images with one initial frame and four-frame updates, or one frame per generation.

    The shared image transfer processor owns control queues, episode resets, and publication of the
    latest generated image. This specialization validates the Cosmos frame cadence before opening
    the camera's generation stream, and joins its camera's views to a session shared with other cameras.

    With :attr:`CosmosModelCfg.generation_mode` ``"single_frame"``, each generation instead makes one image per view
    from the current capture's control, every ``decision_interval`` captures from the latest reset and with
    probability ``augmentation_probability``; the other captures publish the camera's rendered ``rgb``. Only the
    views a capture selects publish a new image, and a capture that selects no view changes nothing.
    """

    def __init__(self, cfg: CosmosTransferProcessorCfg, data: dict[str, ProxyArray]):
        """Validate the Cosmos cadence and initialize the camera's image transfer stream.

        Args:
            cfg: Cosmos camera processor configuration.
            data: Named input buffers; the controls have shape ``(N, H, W, 3)``.

        Raises:
            ValueError: If the configured cadence is incompatible with Cosmos Sim-Transfer, or single-frame
                generation can skip captures but the camera does not render ``rgb`` for them.
        """
        if (cfg.initial_frames, cfg.update_frames) != (1, MAX_CHUNK_FRAMES):
            raise ValueError(f"Cosmos requires initial_frames=1 and update_frames={MAX_CHUNK_FRAMES}.")
        backend = cfg.backend
        self._single_frame = backend.generation_mode == "single_frame"
        if self._single_frame and _skips_captures(cfg) and "rgb" not in data:
            raise ValueError(
                "Single-frame Cosmos publishes the rendered rgb when it skips a capture; add 'rgb' to the Cosmos"
                " processor's inputs (validating the environment configuration adds it)."
            )
        super().__init__(cfg, data)
        # A camera sharing a session takes its views there.
        bind = getattr(self._stream, "bind", None)
        if bind is not None:
            bind(cfg.camera)
        self._capture_index = 0
        self._random = random.Random(self._seeds[0])

    def __call__(self, data: dict[str, ProxyArray], env_mask: wp.array | None = None) -> dict[str, ProxyArray]:
        """Consume controls only when the caller identifies which views have a fresh capture.

        Args:
            data: Named input buffers containing the controls, and the rendered ``rgb`` in single-frame mode.
            env_mask: Boolean mask of captured views. Direct callers must supply it explicitly.

        Returns:
            The latest generated images under the configured output name.

        Raises:
            ValueError: If env_mask is None, as supplied by asynchronous cameras.
        """
        if env_mask is None:
            raise ValueError(
                "Cosmos requires an explicit env_mask; use synchronous rendering"
                " (renderer_cfg.async_rendering=False) for cameras."
            )
        if not self._single_frame:
            return super().__call__(data, env_mask=env_mask)
        return self._generate_single_frame(data, env_mask)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None, env_mask: wp.array | None = None) -> None:
        """Discard the selected views' episode state; in single-frame mode the next capture is a decision point."""
        super().reset(env_ids, env_mask)
        if self._single_frame:
            # Reset is not a hot path; the base class reads the mask on the host too.
            selected = bool(wp.to_torch(env_mask).any()) if env_mask is not None else env_ids is None or len(env_ids)
            if selected:
                self._capture_index = 0

    def _generate_single_frame(self, data: dict[str, ProxyArray], env_mask: wp.array) -> dict[str, ProxyArray]:
        """Publish, for the captured views, one generated image at a decision point, or the rendered rgb otherwise."""
        if self._closed or self._failed:
            raise RuntimeError("Image transfer processor is closed or failed; recreate it.")
        cfg = self._cfg
        frames = data[cfg.inputs[0]].torch
        if tuple(frames.shape) != self._data_dim or frames.dtype != torch.uint8:
            raise ValueError(f"Image transfer controls must be uint8 with shape {self._data_dim}.")
        # Read on the host, as the base class reads it; only the captured views advance.
        rows = [row for row, captured in enumerate(wp.to_torch(env_mask).tolist()) if captured]
        if not rows:
            return dict(self._outputs)
        decision = self._capture_index % cfg.backend.decision_interval == 0
        self._capture_index += 1
        with torch.no_grad():
            if not decision or self._random.random() >= cfg.backend.augmentation_probability:
                rgb = data["rgb"].torch[..., :3]
                if len(rows) == len(self._output):
                    self._output.copy_(rgb)
                else:
                    self._output[rows] = rgb[rows]
                return dict(self._outputs)
            num_views = self._data_dim[0]
            resets = tuple(sorted(self._reset_rows)) if self._started else ()
            seeds = tuple((self._seeds[row] + self._episodes[row] * num_views) % _SEED_MODULUS for row in resets)
            try:
                generated = self._stream.step([frames[row : row + 1] for row in range(num_views)], resets, seeds)
                if len(generated) != num_views or any(
                    tuple(images.shape) != (1, *self._data_dim[1:]) or images.dtype != torch.uint8
                    for images in generated
                ):
                    raise ValueError("Cosmos must return one uint8 RGB image per view.")
            except Exception:
                self._failed = True
                raise
            # The batch generates every view; views not captured now keep their image.
            for row in rows:
                self._output[row].copy_(generated[row][0])
            self._started = True
            self._reset_rows.clear()
        return dict(self._outputs)


def _skips_captures(cfg: CosmosTransferProcessorCfg) -> bool:
    """Whether single-frame generation can skip a capture and publish the rendered rgb instead."""
    return cfg.backend.decision_interval != 1 or cfg.backend.augmentation_probability != 1
