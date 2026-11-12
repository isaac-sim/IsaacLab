# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Cosmos camera processor."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from isaaclab_experimental.image_transfer import ImageTransferProcessorCfg

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
