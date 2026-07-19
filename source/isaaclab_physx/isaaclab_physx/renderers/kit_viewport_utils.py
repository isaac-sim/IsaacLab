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


def _set_kit_camera_view(
    eye: tuple[float, float, float] | list[float],
    target: tuple[float, float, float] | list[float],
    camera_prim_path: str,
):
    """Set a Kit camera pose without loading Isaac Sim's rendering manager."""
    from omni.kit.viewport.utility import get_active_viewport
    from omni.kit.viewport.utility.camera_state import ViewportCameraState
    from pxr import Gf

    viewport_api = get_active_viewport()
    if viewport_api is None:
        raise RuntimeError("No active Kit viewport is available.")

    camera_state = ViewportCameraState(str(camera_prim_path), viewport_api)
    # ``rotate=False`` avoids reading an unauthored
    # ``omni:kit:centerOfInterest`` value from freshly-created perspective
    # cameras. The target call below performs the look-at rotation and authors
    # the center of interest as a side effect.
    camera_state.set_position_world(Gf.Vec3d(*eye), False)
    camera_state.set_target_world(Gf.Vec3d(*target), True)


def set_kit_renderer_camera_view(
    eye: tuple[float, float, float] | list[float],
    target: tuple[float, float, float] | list[float],
    camera_prim_path: str = "/OmniverseKit_Persp",
) -> None:
    """Set camera view for the renderer/viewport camera only.

    This does not broadcast to visualizers.
    """
    try:
        _set_kit_camera_view(eye, target, camera_prim_path)
    except (ImportError, ModuleNotFoundError) as exc:
        logger.debug("[kit_viewport] Renderer camera update skipped (no Kit): %s", exc)
    except Exception as exc:
        logger.warning("[kit_viewport] Renderer camera update failed: %s", exc)
