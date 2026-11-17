# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Foveated camera sensor."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.sensors.camera.camera import Camera

from .virtual_gaze import VirtualGaze

if TYPE_CHECKING:
    from .foveated_camera_cfg import FoveatedCameraCfg


class FoveatedCamera(VirtualGaze, Camera):
    """Camera that samples the scene on a foveated retina instead of a uniform pixel grid.

    The sample layout comes from :class:`fovi.sensing.coords.SamplingCoords`. Constructing this class returns the
    implementation for the renderer selected by :attr:`FoveatedCameraCfg.renderer_cfg`:

    * ``newton_warp``: :class:`~isaaclab_fovi.sensors.foveated_newton_camera.FoveatedNewtonCamera`, which injects
      the foveated rays into the Newton Warp renderer.
    * ``isaac_rtx`` and ``ovrtx``: :class:`~isaaclab_fovi.sensors.foveated_rtx_camera.FoveatedRTXCamera`, which
      authors a lens lookup table on the camera prim.

    Both are subclasses of this class and are imported on first use, so selecting one renderer does not load the
    other's runtime.
    """

    cfg: FoveatedCameraCfg

    def __new__(cls, cfg: FoveatedCameraCfg, *args, **kwargs):
        """Create the implementation that matches the configured renderer.

        Args:
            cfg: Camera configuration with a concrete renderer configuration.

        Raises:
            ValueError: The configured renderer has no foveated implementation.
        """
        if cls is not FoveatedCamera:
            return super().__new__(cls)
        renderer_type = cfg.renderer_cfg.renderer_type
        if renderer_type == "newton_warp":
            from .foveated_newton_camera import FoveatedNewtonCamera

            return super().__new__(FoveatedNewtonCamera)
        if renderer_type in ("isaac_rtx", "ovrtx"):
            from .foveated_rtx_camera import FoveatedRTXCamera

            return super().__new__(FoveatedRTXCamera)
        raise ValueError(
            f"Unsupported foveated renderer {renderer_type!r}; select 'newton_warp', 'isaac_rtx', or 'ovrtx'."
        )
