# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for application-owned image generation applied as camera tensor processors."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.sim import BackendCfg
from isaaclab.utils import configclass
from isaaclab.utils.tensor_processors import TensorProcessorCfg

if TYPE_CHECKING:
    from .processor import DepthControlProcessor, ImageTransferProcessor


@configclass
class DepthControlProcessorCfg(TensorProcessorCfg):
    """Turn the camera's metric depth into the controls of an image transfer: near white, far black."""

    func: type[DepthControlProcessor] | str = "{DIR}.processor:DepthControlProcessor"
    """Depth control processor class."""

    inputs: list[str] = ["distance_to_image_plane"]
    """The depth buffer [m]."""

    outputs: list[str] = ["control"]
    """The control buffer."""

    near: float = MISSING
    """Depth rendered white [m]."""

    far: float = MISSING
    """Depth rendered black [m]. Set the range to match the scene: depth outside it clips to white or black."""


@configclass
class ImageTransferProcessorCfg(TensorProcessorCfg):
    """Generate images from uint8 three-channel controls with an application-owned model.

    Place it after a processor that prepares controls, for example :class:`DepthControlProcessorCfg`, in a
    camera's tensor processors so that it runs once per captured image:

    .. code-block:: python

        CameraCfg(
            data_types=["rgb"],
            tensor_processors=[
                DepthControlProcessorCfg(near=0.1, far=10.0),
                ImageTransferProcessorCfg(backend=MyModelCfg(), update_frames=4),
            ],
        )

    The output holds the latest generated image between chunks. Resets discard the reset views'
    queued controls and episode state.
    """

    func: type[ImageTransferProcessor] | str = "{DIR}.processor:ImageTransferProcessor"
    """Image transfer processor class."""

    inputs: list[str] = ["control"]
    """The control buffer."""

    outputs: list[str] = ["rgb"]
    """The generated image buffer."""

    backend: BackendCfg = MISSING
    """Model configuration. Equal configurations share one model through the simulation context; the
    constructed model must implement :class:`ImageTransferModel`."""

    seed: int | list[int] = 0
    """Initial seed of view 0. View ``i`` starts from ``seed + i``, or from ``seed[i]`` when a list gives one seed
    per view, for example per environment. Each reset advances a view's seed by the number of views, modulo
    ``2**31``."""

    initial_frames: int = 1
    """Captures required for the first chunk of each episode."""

    update_frames: int = 1
    """Captures required for subsequent chunks."""

    max_pending_frames: int = 8
    """Maximum queued captures per view. A capture beyond it raises an error when the processor next checks for a
    complete chunk, at most ``max(initial_frames, update_frames)`` calls later."""
