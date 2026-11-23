# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Cosmos camera processor."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab_cosmos_server.protocol import MAX_CHUNK_FRAMES

from isaaclab.utils import configclass

from ..image_transfer import ImageTransferProcessorCfg

if TYPE_CHECKING:
    from .cosmos_model_cfg import CosmosModelCfg
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
    :func:`~isaaclab_contrib.sensors.cosmos.cosmos_camera` sets for a camera whose size is not a Cosmos canvas."""

    def validate_config(self):
        """Make ``rgb`` an input exactly when single-frame generation can skip captures.

        The camera renders a processor's inputs, and skipped captures publish the rendered ``rgb``; the processor
        reads no other input than its control. The generation settings can be overridden until validation, for
        example with ``env.cosmos.augmentation_probability=0.5``, so the inputs follow the settings at each
        validation.
        """
        inputs = [name for index, name in enumerate(self.inputs) if index == 0 or name != "rgb"]
        if publishes_rendered_rgb(self.backend) and "rgb" not in inputs:
            inputs.append("rgb")
        self.inputs = inputs


def publishes_rendered_rgb(backend: CosmosModelCfg) -> bool:
    """Whether single-frame generation can skip a capture and publish the rendered ``rgb`` instead."""
    return getattr(backend, "generation_mode", None) == "single_frame" and (
        backend.decision_interval != 1 or backend.augmentation_probability != 1
    )
