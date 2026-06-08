# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit / Omniverse viewport helpers (Isaac Sim specific).

These live in :mod:`isaaclab_physx` so :class:`~isaaclab.sim.SimulationContext` stays
backend-agnostic.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def set_kit_renderer_camera_view(
    eye: tuple[float, float, float] | list[float],
    target: tuple[float, float, float] | list[float],
    camera_prim_path: str = "/OmniverseKit_Persp",
) -> None:
    """Set camera view for the renderer/viewport camera only.

    This does not broadcast to visualizers.
    """
    try:
        from isaacsim.core.rendering_manager import ViewportManager

        ViewportManager.set_camera_view(
            str(camera_prim_path),
            eye=list(eye),
            target=list(target),
        )
    except (ImportError, ModuleNotFoundError) as exc:
        logger.debug("[kit_viewport] Renderer camera update skipped (no Kit): %s", exc)
    except Exception as exc:
        logger.warning("[kit_viewport] Renderer camera update failed: %s", exc)
