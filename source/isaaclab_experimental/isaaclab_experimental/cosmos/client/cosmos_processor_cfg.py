# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Cosmos camera processor."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from ...image_transfer import ImageTransferProcessorCfg
from .._protocol import MAX_CHUNK_FRAMES

if TYPE_CHECKING:
    from .cosmos_processor import CosmosTransferProcessor


@configclass
class CosmosTransferProcessorCfg(ImageTransferProcessorCfg):
    """Generate one initial frame, then four frames per Cosmos Sim-Transfer chunk."""

    func: type[CosmosTransferProcessor] | str = "{DIR}.cosmos_processor:CosmosTransferProcessor"
    """The Cosmos camera processor."""

    initial_frames: int = 1
    """Captures required for the first chunk of an episode."""

    update_frames: int = MAX_CHUNK_FRAMES
    """Captures required for subsequent chunks."""

    camera: str = ""
    """Scene name of the processor's camera, one of :attr:`CosmosModelCfg.cameras` when cameras share a session.
    Defaults to empty, for a camera with its own session."""

    image_size: tuple[int, int] | None = None
    """Height and width of the published images. Defaults to None, the control's size. Another size publishes the
    center crop of the control's canvas with that aspect ratio, area-resized, as
    :func:`~isaaclab_experimental.cosmos.cosmos_camera` sets for a camera whose size is not a Cosmos canvas. The
    processor resizes the images itself, so images a shared session generates after this processor ran still reach
    the camera's published output."""

    def validate_config(self):
        """Have the camera render ``rgb`` when single-frame generation can skip captures and publish it instead.

        The camera renders the inputs no earlier processor produces, so the rendered image reaches this processor
        only when it is an input; it is added here because the generation settings can change after this
        configuration is built, for example with ``env.cosmos.augmentation_probability=0.5``.
        """
        backend = self.backend
        if (
            getattr(backend, "generation_mode", "autoregressive") == "single_frame"
            and (backend.decision_interval != 1 or backend.augmentation_probability != 1)
            and "rgb" not in self.inputs
        ):
            self.inputs = [*self.inputs, "rgb"]
