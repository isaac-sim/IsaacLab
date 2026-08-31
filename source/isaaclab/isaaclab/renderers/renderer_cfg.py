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

Accepts boolean spellings: ``0``/``false``/``no``/``off`` or ``1``/``true``/``yes``/``on``. Set it
to exercise the asynchronous path without naming a camera that a given task may not define.
"""

_FALSE_SPELLINGS = ("0", "false", "no", "off")
_TRUE_SPELLINGS = ("1", "true", "yes", "on")


@configclass
class RendererCfg:
    """Configuration for a renderer."""

    class_type: type[BaseRenderer] | str | None = None
    """Renderer implementation class. Concrete configs must set this field."""

    renderer_type: str = "default"

    async_rendering: bool = False
    """Trade one frame of camera latency for pipelined rendering. Defaults to False (synchronous).

    ``False`` renders synchronously: each render completes before the step returns. ``True``
    pipelines rendering one frame deep, so rendering overlaps the next step's simulation and Python
    work and camera outputs describe the simulation state from one step earlier. ``0``/``1`` are
    accepted as shorthand, as are the usual string spellings through the environment variable;
    frame counts above one are rejected until multi-frame latency lands.

    Only the OVRTX renderer implements this; other renderers warn and render synchronously. One
    frame is also the most its ovstage scene-ownership path can pipeline: ovstage retains a single
    committed snapshot that renders read in place, so each frame's scene writes must drain the
    render still in flight. Deeper queues on the legacy path are possible future work once they are
    worth their review surface. :data:`ASYNC_RENDERING_ENV_VAR` overrides this value.
    """


def _parse_async_rendering_enabled(value: bool | int | str) -> bool:
    """Normalize an :attr:`RendererCfg.async_rendering` value to a flag.

    Args:
        value: A boolean, ``0``/``1``, or a string spelling of either.

    Returns:
        Whether asynchronous rendering is enabled.

    Raises:
        ValueError: If ``value`` is not a recognized boolean spelling. Frame counts above one are
            rejected explicitly: multi-frame latency is not supported yet.
    """
    # ``bool`` is a subclass of ``int``, so it has to be matched before the integer case.
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"async rendering is a flag; multi-frame latency ({value}) is not supported yet")
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _FALSE_SPELLINGS:
            return False
        if text in _TRUE_SPELLINGS:
            return True
    raise ValueError(f"async rendering expects a boolean spelling, got {value!r}")


def async_rendering_enabled_from_env() -> bool | None:
    """Return the :data:`ASYNC_RENDERING_ENV_VAR` override.

    Returns:
        The override flag, or ``None`` when the variable is unset, empty, or unparsable.
    """
    raw = os.environ.get(ASYNC_RENDERING_ENV_VAR)
    if raw is None or raw.strip() == "":
        return None
    try:
        return _parse_async_rendering_enabled(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; expected a boolean spelling.", ASYNC_RENDERING_ENV_VAR, raw)
        return None


def resolve_async_rendering_enabled(cfg: RendererCfg) -> bool:
    """Return whether ``cfg`` asks for asynchronous rendering, honoring :data:`ASYNC_RENDERING_ENV_VAR`.

    Args:
        cfg: Renderer configuration to read :attr:`RendererCfg.async_rendering` from.

    Returns:
        Whether asynchronous rendering is enabled.

    Raises:
        ValueError: If :attr:`RendererCfg.async_rendering` is not a recognized boolean spelling.
    """
    override = async_rendering_enabled_from_env()
    if override is not None:
        return override
    return _parse_async_rendering_enabled(cfg.async_rendering)


def warn_unsupported_async_rendering(cfg: RendererCfg, renderer_name: str) -> None:
    """Warn when ``cfg`` asks for asynchronous rendering but the renderer only renders synchronously.

    Renderers that do not pipeline call this during construction so a requested-but-ignored
    :attr:`RendererCfg.async_rendering` is visible rather than silently dropped.

    Args:
        cfg: Renderer configuration to read :attr:`RendererCfg.async_rendering` from.
        renderer_name: Renderer to name in the warning, e.g. ``"newton_warp"``.
    """
    if resolve_async_rendering_enabled(cfg):
        logger.warning(
            "Asynchronous rendering is not implemented for the %s renderer; rendering synchronously."
            " Only the OVRTX renderer pipelines renders.",
            renderer_name,
        )
