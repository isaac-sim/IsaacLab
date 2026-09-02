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

Accepts boolean spellings: ``0``/``false``/``no``/``off`` or ``1``/``true``/``yes``/``on``. Any
other value raises ``ValueError``. Set it to exercise the asynchronous path without naming a
camera that a given task may not define.
"""


@configclass
class RendererCfg:
    """Configuration for a renderer."""

    class_type: type[BaseRenderer] | str | None = None
    """Renderer implementation class. Concrete configs must set this field."""

    renderer_type: str = "default"

    async_rendering: bool = False
    """Trade one frame of camera latency for pipelined rendering. Defaults to False (synchronous).

    When enabled, rendering overlaps the next step's simulation and Python work, and camera
    outputs describe the simulation state from one step earlier. Only the OVRTX renderer
    implements this. Other renderers warn and render synchronously.
    :data:`ASYNC_RENDERING_ENV_VAR` overrides this value.
    """


# TODO: IsaacLab hand-rolls parsing like this for several boolean environment variables and
# configuration values. Replace these copies with one shared argument-parsing helper.
def _parse_flag(value: bool | int | str) -> bool:
    """Normalize a flag value, accepting the usual boolean spellings."""
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    raise ValueError(
        f"async rendering expects a boolean spelling, got {value!r}. Frame counts above one are not supported yet."
    )


def resolve_async_rendering_enabled(cfg: RendererCfg) -> bool:
    """Return whether ``cfg`` asks for asynchronous rendering, honoring :data:`ASYNC_RENDERING_ENV_VAR`.

    Args:
        cfg: Renderer configuration to read :attr:`RendererCfg.async_rendering` from.

    Returns:
        Whether asynchronous rendering is enabled.

    Raises:
        ValueError: If the configuration value or the environment variable is not a recognized
            boolean spelling. The configuration value is validated even when the environment
            variable overrides it, so an invalid configuration fails in every environment.
    """
    enabled = _parse_flag(cfg.async_rendering)
    override = os.environ.get(ASYNC_RENDERING_ENV_VAR, "").strip()
    return _parse_flag(override) if override else enabled


def warn_unsupported_async_rendering(cfg: RendererCfg, renderer_name: str) -> None:
    """Warn when ``cfg`` asks for asynchronous rendering but the renderer only renders synchronously.

    Renderers that do not pipeline call this during construction so a requested-but-ignored
    :attr:`RendererCfg.async_rendering` is visible rather than silently dropped.

    Args:
        cfg: Renderer configuration to read :attr:`RendererCfg.async_rendering` from.
        renderer_name: Renderer to name in the warning, e.g. ``"newton_warp"``.

    Raises:
        ValueError: If the configuration value or the environment variable is not a recognized
            boolean spelling. Renderers that ignore the setting still refuse an invalid value.
    """
    if resolve_async_rendering_enabled(cfg):
        logger.warning(
            "Asynchronous rendering is not implemented for the %s renderer; rendering synchronously."
            " Only the OVRTX renderer pipelines renders.",
            renderer_name,
        )
