# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for renderers."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .base_renderer import BaseRenderer

logger = logging.getLogger(__name__)

ASYNC_RENDERING_ENV_VAR = "ISAAC_LAB_ASYNC_RENDERING"
"""Environment variable overriding :attr:`RendererCfg.async_rendering` for every renderer.

Accepts the same values as the config field: a boolean spelling (``0``/``false``/``no``/``off`` or
``1``/``true``/``yes``/``on``) or a frame count. Set it to exercise the asynchronous path without
naming a camera that a given task may not define.
"""

_FALSE_SPELLINGS = ("0", "false", "no", "off")
_TRUE_SPELLINGS = ("1", "true", "yes", "on")


@configclass
class RendererCfg:
    """Configuration for a renderer."""

    class_type: type[BaseRenderer] | str | None = None
    """Renderer implementation class. Concrete configs must set this field."""

    renderer_type: str = "default"

    async_rendering: bool | int = False
    """Frames of camera latency traded for pipelined rendering. Defaults to False (synchronous).

    ``False`` and ``0`` render synchronously: each render completes before the step returns. ``True``
    means one frame. An integer ``n > 0`` keeps ``n`` renders in flight, so camera outputs describe the
    simulation state from ``n`` steps earlier; larger values overlap more simulation and Python work
    with rendering, raising throughput at the cost of staler camera data.

    Only the OVRTX renderer implements this; other renderers warn and render synchronously.
    :data:`ASYNC_RENDERING_ENV_VAR` overrides this value.
    """


def _parse_async_rendering_frames(value: bool | int | str) -> int:
    """Normalize an :attr:`RendererCfg.async_rendering` value to a frame count.

    Args:
        value: A boolean, a frame count, or a string spelling of either.

    Returns:
        Frames of render latency, where ``0`` means synchronous rendering.

    Raises:
        ValueError: If ``value`` is not a boolean or a frame count ``>= 0``.
    """
    # ``bool`` is a subclass of ``int``, so it has to be matched before the integer case.
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        frames = value
    else:
        text = value.strip().lower()
        if text in _FALSE_SPELLINGS:
            return 0
        if text in _TRUE_SPELLINGS:
            return 1
        frames = int(text)
    if frames < 0:
        raise ValueError(f"async rendering expects a frame count >= 0, got {frames}")
    return frames


def async_rendering_frames_from_env() -> int | None:
    """Return the :data:`ASYNC_RENDERING_ENV_VAR` override.

    Returns:
        Frames of render latency, or ``None`` when the variable is unset, empty, or unparsable.
    """
    raw = os.environ.get(ASYNC_RENDERING_ENV_VAR)
    if raw is None or raw.strip() == "":
        return None
    try:
        return _parse_async_rendering_frames(raw)
    except ValueError:
        logger.warning(
            "Ignoring invalid %s=%r; expected a boolean or a frame count >= 0.", ASYNC_RENDERING_ENV_VAR, raw
        )
        return None


def resolve_async_rendering_frames(cfg: RendererCfg) -> int:
    """Return the frames of render latency ``cfg`` asks for, honoring :data:`ASYNC_RENDERING_ENV_VAR`.

    Args:
        cfg: Renderer configuration to read :attr:`RendererCfg.async_rendering` from.

    Returns:
        Frames of render latency, where ``0`` means synchronous rendering.

    Raises:
        ValueError: If :attr:`RendererCfg.async_rendering` is not a boolean or a frame count ``>= 0``.
    """
    override = async_rendering_frames_from_env()
    if override is not None:
        return override
    return _parse_async_rendering_frames(cfg.async_rendering)


def warn_unsupported_async_rendering(cfg: RendererCfg, renderer_name: str) -> None:
    """Warn when ``cfg`` asks for asynchronous rendering but the renderer only renders synchronously.

    Renderers that do not pipeline call this during construction so a requested-but-ignored
    :attr:`RendererCfg.async_rendering` is visible rather than silently dropped.

    Args:
        cfg: Renderer configuration to read :attr:`RendererCfg.async_rendering` from.
        renderer_name: Renderer to name in the warning, e.g. ``"newton_warp"``.
    """
    if resolve_async_rendering_frames(cfg) > 0:
        logger.warning(
            "Asynchronous rendering is not implemented for the %s renderer; rendering synchronously."
            " Only the OVRTX renderer pipelines renders.",
            renderer_name,
        )
