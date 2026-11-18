# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cosmos image generation applied to camera images."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch
import warp as wp

from ...image_transfer import ImageTransferProcessor
from .._protocol import MAX_CHUNK_FRAMES

if TYPE_CHECKING:
    from isaaclab.utils.warp import ProxyArray

    from .cosmos_processor_cfg import CosmosTransferProcessorCfg


class CosmosTransferProcessor(ImageTransferProcessor):
    """Generate Cosmos camera images with one initial frame and four-frame updates, or one frame per generation.

    The shared image transfer processor owns control queues, episode resets, and publication of the
    latest generated image. This specialization validates the Cosmos frame cadence before opening
    the camera's generation stream, and joins its camera's views to a session shared with other cameras.

    With :attr:`CosmosModelCfg.generation_mode` ``"single_frame"``, each generation instead makes one image per view
    from the current capture's control, at every ``decision_interval``-th capture of the camera and with
    probability ``augmentation_probability``; the other captures publish the camera's rendered ``rgb``. Resets do
    not move this schedule, and only the views a capture selects publish a new image.
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

    def _generate_single_frame(self, data: dict[str, ProxyArray], env_mask: wp.array) -> dict[str, ProxyArray]:
        """Publish, for the captured views, one generated image at a decision point, or the rendered rgb otherwise."""
        if self._closed or self._failed:
            raise RuntimeError("Image transfer processor is closed or failed; recreate it.")
        cfg = self._cfg
        frames = data[cfg.inputs[0]].torch
        if tuple(frames.shape) != self._data_dim or frames.dtype != torch.uint8:
            raise ValueError(f"Image transfer controls must be uint8 with shape {self._data_dim}.")
        # Views the capture does not select keep their image; the mask stays on the device.
        selected = wp.to_torch(env_mask).to(self._output.device).view(-1, 1, 1, 1)
        decision = self._capture_index % cfg.backend.decision_interval == 0
        self._capture_index += 1
        with torch.no_grad():
            if not decision or self._random.random() >= cfg.backend.augmentation_probability:
                torch.where(selected, data["rgb"].torch[..., :3], self._output, out=self._output)
                return dict(self._outputs)
            num_views = self._data_dim[0]
            resets, seeds = self._take_resets()
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
            images = torch.cat(generated).to(self._output.device)
            torch.where(selected, images, self._output, out=self._output)
            self._started = True
        return dict(self._outputs)


def _skips_captures(cfg: CosmosTransferProcessorCfg) -> bool:
    """Whether single-frame generation can skip a capture and publish the rendered rgb instead."""
    return cfg.backend.decision_interval != 1 or cfg.backend.augmentation_probability != 1
