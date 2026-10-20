# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cosmos image generation applied to camera observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_experimental.image_transfer import ImageTransferModifier

if TYPE_CHECKING:
    from isaaclab.sensors import SensorBase

    from .cosmos_modifier_cfg import CosmosTransferModifierCfg


class CosmosTransferModifier(ImageTransferModifier):
    """Generate Cosmos camera images with one initial frame and four-frame updates.

    The shared image transfer modifier owns control queues, episode resets, and publication of the
    latest generated image. This specialization validates the Cosmos frame cadence before opening
    the camera's generation stream.
    """

    def __init__(self, cfg: CosmosTransferModifierCfg, data_dim: tuple[int, ...], device: str):
        """Validate the Cosmos cadence and initialize the camera's image transfer stream.

        Args:
            cfg: Cosmos camera modifier configuration.
            data_dim: Control shape ``(N, H, W, 3)``.
            device: Device of the controls and generated output.

        Raises:
            ValueError: If the configured cadence is incompatible with Cosmos Sim-Transfer.
        """
        if (cfg.initial_frames, cfg.update_frames) != (1, 4):
            raise ValueError("Cosmos requires initial_frames=1 and update_frames=4.")
        super().__init__(cfg, data_dim, device)

    def bind_sensor(self, sensor: SensorBase) -> None:
        """Queue controls for the views ``sensor`` captures; a camera sharing a session also takes its views there."""
        super().bind_sensor(sensor)
        bind = getattr(self._stream, "bind_sensor", None)
        if bind is not None:
            bind(sensor)
