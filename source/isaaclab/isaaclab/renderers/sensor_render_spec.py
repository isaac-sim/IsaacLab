# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable description of a tiled sensor passed to render backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..sensors.camera.camera_cfg import CameraCfg


@dataclass(frozen=True)
class SensorRenderSpec:
    """Stable inputs for :meth:`~isaaclab.renderers.base_renderer.BaseRenderer.create_render_data`.

    Backends use this instead of holding a reference to the sensor instance,
    avoiding circular dependencies between sensors and render data.

    Args:
        cfg: Sensor configuration.
        device: Torch device string (e.g. ``"cuda:0"``) used by GPU annotators and Warp.
        num_instances: Number of tiled sensor instances (environments).
        prim_paths: Absolute paths of the authored sensor prims. When the renderer clones
            environments internally, this may contain only the source prim path; the renderer
            resolves its logical per-environment paths during registration.
        view_count: Number of logical sensor instances in the sensor view.
    """

    cfg: CameraCfg
    device: str
    num_instances: int
    prim_paths: tuple[str, ...]
    view_count: int
