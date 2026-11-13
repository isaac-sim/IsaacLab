# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for application-owned image generation applied as camera tensor processors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.sensors.camera import CAPTURE_ENV_MASK_NAME
from isaaclab.sim import BackendCfg
from isaaclab.utils import configclass
from isaaclab.utils.tensor_processors import TensorProcessorCfg

if TYPE_CHECKING:
    import torch

    from .processor import FunctionProcessor, ImageTransferProcessor


@configclass
class FunctionProcessorCfg(TensorProcessorCfg):
    """Apply a Torch function to one camera buffer, for example to prepare controls:

    .. code-block:: python

        FunctionProcessorCfg(
            function=depth_to_control,
            params={"near": 0.1, "far": 10.0},
            inputs=["distance_to_image_plane"],
            outputs=["control"],
        )
    """

    func: type[FunctionProcessor] | str = "{DIR}.processor:FunctionProcessor"
    """Function processor class."""

    function: Callable[..., torch.Tensor] | str = MISSING
    """Function applied to the input tensor, with :attr:`params` as keyword arguments, or its import string."""

    inputs: list[str] = MISSING
    """The input buffer name."""

    outputs: list[str] = MISSING
    """The output buffer name."""


@configclass
class ImageTransferProcessorCfg(TensorProcessorCfg):
    """Generate images from uint8 three-channel controls with an application-owned model.

    Place it after a processor that prepares controls, for example :func:`depth_to_control`, in a
    camera's tensor processors so that it runs once per captured image:

    .. code-block:: python

        CameraCfg(
            data_types=["rgb"],
            tensor_processors=[
                FunctionProcessorCfg(
                    function=depth_to_control,
                    params={"near": 0.1, "far": 10.0},
                    inputs=["distance_to_image_plane"],
                    outputs=["control"],
                ),
                ImageTransferProcessorCfg(backend=MyModelCfg(), update_frames=4),
            ],
        )

    The output holds the latest generated image between chunks. Resets discard the reset views'
    queued controls and episode state.
    """

    func: type[ImageTransferProcessor] | str = "{DIR}.processor:ImageTransferProcessor"
    """Image transfer processor class."""

    inputs: list[str] = ["control", CAPTURE_ENV_MASK_NAME]
    """The control buffer, then the camera's capture mask."""

    outputs: list[str] = ["rgb"]
    """The generated image buffer."""

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
