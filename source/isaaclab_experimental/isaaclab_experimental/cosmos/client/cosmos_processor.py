# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cosmos image generation applied to camera images."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_experimental.image_transfer import ImageTransferProcessor

from .._protocol import MAX_CHUNK_FRAMES

if TYPE_CHECKING:
    from isaaclab.sensors import SensorTensorProcessorArgs
    from isaaclab.utils.warp import ProxyArray

    from .cosmos_processor_cfg import CosmosTransferProcessorCfg


class CosmosTransferProcessor(ImageTransferProcessor):
    """Generate Cosmos camera images with one initial frame and four-frame updates.

    The shared image transfer processor owns control queues, episode resets, and publication of the
    latest generated image. This specialization validates the Cosmos frame cadence before opening
    the camera's generation stream, and joins its camera's views to a session shared with other cameras.
    """

    def __init__(self, cfg: CosmosTransferProcessorCfg, data: dict[str, ProxyArray]):
        """Validate the Cosmos cadence and initialize the camera's image transfer stream.

        Args:
            cfg: Cosmos camera processor configuration.
            data: Named input buffers; the controls have shape ``(N, H, W, 3)``.

        Raises:
            ValueError: If the configured cadence is incompatible with Cosmos Sim-Transfer.
        """
        if (cfg.initial_frames, cfg.update_frames) != (1, MAX_CHUNK_FRAMES):
            raise ValueError(f"Cosmos requires initial_frames=1 and update_frames={MAX_CHUNK_FRAMES}.")
        super().__init__(cfg, data)
        # A camera sharing a session takes its views there.
        bind = getattr(self._stream, "bind", None)
        if bind is not None:
            bind(cfg.camera)

    def __call__(
        self, data: dict[str, ProxyArray], args: SensorTensorProcessorArgs | None = None
    ) -> dict[str, ProxyArray]:
        """Consume controls only when the caller identifies which views have a fresh capture.

        Args:
            data: Named input buffers containing the controls.
            args: Arguments whose ``env_mask`` marks the captured views. Direct callers must supply it.

        Returns:
            The latest generated images under the configured output name.

        Raises:
            ValueError: If ``args`` or its ``env_mask`` is None, as supplied by asynchronous cameras.
        """
        if args is None or args.env_mask is None:
            raise ValueError(
                "Cosmos requires an explicit env_mask; use synchronous rendering"
                " (renderer_cfg.async_rendering=False) for cameras."
            )
        return super().__call__(data, args=args)
