# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for application-owned image generation applied as a modifier."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.sim import BackendCfg
from isaaclab.utils import configclass
from isaaclab.utils.modifiers import ModifierCfg

if TYPE_CHECKING:
    from .modifier import ImageTransferModifier


@configclass
class ImageTransferModifierCfg(ModifierCfg):
    """Generate images from uint8 three-channel controls with an application-owned model.

    Place it after a modifier that prepares controls, for example :func:`depth_to_control`, in a
    camera modifier chain so that it runs once per captured image:

    .. code-block:: python

        CameraCfg(
            data_types=["rgb"],
            modifiers={
                "distance_to_image_plane": [
                    ModifierCfg(func=depth_to_control, params={"near": 0.1, "far": 10.0}),
                    ImageTransferModifierCfg(backend=MyModelCfg(), update_frames=4),
                ]
            },
        )

    The output holds the latest generated image between chunks. Resets discard the reset views'
    queued controls and episode state.
    """

    func: type[ImageTransferModifier] | str = "{DIR}.modifier:ImageTransferModifier"
    """Image transfer modifier class."""

    backend: BackendCfg = MISSING
    """Model configuration. Equal configurations share one model through the simulation context; the
    constructed model must implement :class:`ImageTransferModel`."""

    seed: int = 0
    """Initial seed of view 0. View ``i`` starts from ``seed + i``; each reset advances a view's seed by
    the number of views, modulo ``2**31``."""

    initial_frames: int = 1
    """Captures required for the first chunk of each episode."""

    update_frames: int = 1
    """Captures required for subsequent chunks."""

    max_pending_frames: int = 8
    """Maximum queued captures per view before backpressure raises an error."""

    output: str = "rgb"
    """Camera output produced from the controls."""
