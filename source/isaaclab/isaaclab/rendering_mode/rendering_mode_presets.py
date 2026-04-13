# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stable entry point for built-in RTX presets.

Preset tables are defined in :mod:`isaaclab_physx.rendering.rtx_rendering_mode_presets` and loaded here
so ``isaaclab.rendering_mode`` keeps a single public API.
"""

from __future__ import annotations

from typing import Any


def get_rendering_mode_preset(preset_name: str) -> dict[str, Any]:
    """Return a deep copy of the requested built-in rendering mode preset.

    Raises:
        ImportError: If ``isaaclab_physx`` is not installed (RTX presets live in that package).
    """
    try:
        from isaaclab_physx.rendering.rtx_rendering_mode_presets import get_builtin_rtx_rendering_mode_preset
    except ImportError as e:
        raise ImportError(
            "RTX rendering mode presets are provided by isaaclab_physx. "
            "Install with: pip install 'isaaclab[physx]' or install the isaaclab_physx package."
        ) from e
    return get_builtin_rtx_rendering_mode_preset(preset_name)
