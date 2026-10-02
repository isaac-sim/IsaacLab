# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for sensor post-processors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

from ...renderers.output_contract import RenderBufferSpec
from ...utils import configclass

if TYPE_CHECKING:
    from .post_processor import CameraPostProcessorContext, SensorPostProcessor


@configclass
class SensorPostProcessorCfg:
    """Configure a factory that creates independent state for each processing chain.

    ``func(cfg, context)`` resolves configuration before renderer setup and returns a
    :class:`~isaaclab.sensors.post_processing.SensorPostProcessor`, or ``None`` to disable this
    operation. Static ``inputs`` declare potential renderer requirements needed before simulation
    startup (e.g. HDR). The returned processor declares the actual inputs and outputs after discovery.
    """

    func: Callable[[SensorPostProcessorCfg, CameraPostProcessorContext], SensorPostProcessor | None] = MISSING
    inputs: dict[str, RenderBufferSpec] = {}
    outputs: dict[str, RenderBufferSpec] = {}
    params: dict[str, Any] = {}
