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
from isaaclab_cosmos_server.protocol import MAX_CHUNK_FRAMES

from isaaclab.utils.warp import ProxyArray

from ..image_transfer import ImageTransferProcessor
from .cosmos_processor_cfg import publishes_rendered_rgb

if TYPE_CHECKING:
    from .cosmos_processor_cfg import CosmosTransferProcessorCfg


class CosmosTransferProcessor(ImageTransferProcessor):
    """Generate Cosmos camera images with one initial frame and four-frame updates, or one frame per generation.

    The shared image transfer processor owns control queues, episode resets, and publication of the
    latest generated image. This specialization validates the Cosmos frame cadence before opening
    the camera's generation stream, and joins its camera's views to a session shared with other cameras.

    With :attr:`CosmosClientCfg.generation_mode` ``"single_frame"``, each generation instead makes one image per view
    from the current capture's control, at every ``decision_interval``-th capture of the camera and with
    probability ``augmentation_probability``; the other captures publish the camera's rendered ``rgb``. Resets do
    not move this schedule, and only the views a capture selects publish a new image.

    With :attr:`CosmosTransferProcessorCfg.image_size`, the processor keeps the images at the control's size and
    publishes their center crop at that size.
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
        if publishes_rendered_rgb(backend) and "rgb" not in data:
            raise ValueError(
                "Single-frame Cosmos publishes the rendered rgb when it skips a capture; add 'rgb' to the Cosmos"
                " processor's inputs (validating the environment configuration adds it)."
            )
        super().__init__(cfg, data)
        # Images land in the control-sized output; a camera of another size publishes a resized copy of it.
        self._resized: torch.Tensor | None = None
        if cfg.image_size is not None and tuple(cfg.image_size) != self._data_dim[1:3]:
            shape = (self._data_dim[0], *cfg.image_size, 3)
            resized = wp.zeros(shape, dtype=wp.uint8, device=str(self._output.device))
            self._outputs = {cfg.outputs[0]: ProxyArray(resized)}
            self._resized = self._outputs[cfg.outputs[0]].torch
        # A camera sharing a session takes its views there, and gets its images when the session generates them.
        bind = getattr(self._stream, "bind", None)
        if bind is not None:
            bind(cfg.camera, self._publish)
        self._capture_index = 0
        self._random = random.Random(self._seeds[0])
        # Whether reset() ran since the resets were last read, so a generation reads them only after one.
        self._reset_called = False

    def __call__(self, data: dict[str, ProxyArray], env_mask: wp.array | None = None) -> dict[str, ProxyArray]:
        """Queue or generate from the controls of the captured views.

        Args:
            data: Named input buffers containing the controls, and the rendered ``rgb`` in single-frame mode.
            env_mask: Boolean mask of the views captured in this call.

        Returns:
            The latest generated images under the configured output name.

        Raises:
            ValueError: If env_mask is None, as supplied by asynchronous cameras.
        """
        if not self._single_frame:
            return super().__call__(data, env_mask=env_mask)
        return self._generate_single_frame(data, env_mask)

    def _generate_single_frame(self, data: dict[str, ProxyArray], env_mask: wp.array) -> dict[str, ProxyArray]:
        """Publish, for the captured views, one generated image at a decision point, or the rendered rgb otherwise."""
        cfg = self._cfg
        frames = self._controls(data, env_mask)
        # Views the capture does not select keep their image; the mask stays on the device.
        selected = wp.to_torch(env_mask).to(self._output.device).view(-1, 1, 1, 1)
        decision = self._capture_index % cfg.backend.decision_interval == 0
        self._capture_index += 1
        with torch.no_grad():
            if not decision or self._random.random() >= cfg.backend.augmentation_probability:
                torch.where(selected, data["rgb"].torch[..., :3], self._output, out=self._output)
                self._resize()
                return dict(self._outputs)
            num_views = self._data_dim[0]
            # Episode numbers let the prompt list follow episodes that ended between two generations.
            resets, seeds, episodes = self._take_resets()
            controls = [frames[row : row + 1] for row in range(num_views)]
            try:
                generated = self._stream.step(controls, resets, seeds, episodes=episodes)
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
            self._resize()
        return dict(self._outputs)

    def _publish(self, generated: list[torch.Tensor] | None, chunk: list[int] | None = None) -> None:
        """Publish each view's last generated image, at :attr:`CosmosTransferProcessorCfg.image_size`.

        A session shared with other cameras returns None from a step and calls this without ``chunk`` when it
        generates that chunk.
        """
        if generated is None:
            return
        super()._publish(generated, chunk if chunk is not None else [len(images) for images in generated])
        self._resize()

    def _reset_views(self, selected: torch.Tensor) -> None:
        """Discard the selected views' queued controls and episode state, and publish them black."""
        super()._reset_views(selected)
        self._reset_called = True
        if self._resized is not None:
            self._resized.masked_fill_(selected.view(-1, 1, 1, 1), 0)

    def _take_resets(self) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Return the views that started an episode since the last generation, with their seeds and episode
        numbers, and clear them. A view that reset several times is listed once, with its latest episode.

        This reads the state on the host after a reset, so call it only where the generation itself waits for the
        result.
        """
        if not self._reset_called:
            return (), (), ()
        self._reset_called = False
        resets, episodes = torch.stack((self._resets.long(), self._episodes)).tolist()
        self._resets.zero_()
        rows = tuple(row for row, reset in enumerate(resets) if reset)
        return rows, self._episode_seeds(rows, episodes), tuple(episodes[row] for row in rows)

    def _resize(self) -> None:
        """Publish the control-sized images at the configured image size."""
        if self._resized is not None:
            height, width = self._cfg.image_size
            self._resized.copy_(_center_crop_resize(self._output, width=width, height=height))


def _center_crop_resize(data: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Cut the camera's original view out of the canvas's center and area-resize it to the camera's size."""
    _, rows, cols, _ = data.shape
    if cols * height > rows * width:
        crop = rows * width // height
        data = data[:, :, (cols - crop) // 2 : (cols - crop) // 2 + crop]
    else:
        crop = cols * height // width
        data = data[:, (rows - crop) // 2 : (rows - crop) // 2 + crop]
    pixels = torch.nn.functional.interpolate(data.permute(0, 3, 1, 2).float(), size=(height, width), mode="area")
    return pixels.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
