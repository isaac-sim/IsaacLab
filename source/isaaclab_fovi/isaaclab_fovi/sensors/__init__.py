# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera sensors and simulator-independent projection helpers.

Backend classes are imported on demand so Newton and pure geometry callers
do not load the RTX runtime.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import warp as wp

    from .foveated_newton_tiled_camera import FoveatedNewtonTiledCamera
    from .foveated_rays import compute_foveated_ray_array
    from .foveated_rtx_tiled_camera import FoveatedRTXTiledCamera
    from .foveated_tiled_camera_cfg import FoveatedTiledCameraCfg

__all__ = [
    "FoveatedNewtonTiledCamera",
    "FoveatedRTXTiledCamera",
    "FoveatedTiledCamera",
    "FoveatedTiledCameraCfg",
    "compute_foveated_ray_array",
]


def FoveatedTiledCamera(
    cfg: FoveatedTiledCameraCfg,
) -> FoveatedNewtonTiledCamera | FoveatedRTXTiledCamera:
    """Construct a foveated camera for the explicitly selected renderer.

    Args:
        cfg: Camera configuration with a concrete renderer configuration.

    Raises:
        ValueError: The renderer is not supported.
        NotImplementedError: Non-tiled Isaac RTX rendering is requested.
    """
    renderer_type = cfg.renderer_cfg.renderer_type
    if renderer_type == "newton_warp":
        from .foveated_newton_tiled_camera import FoveatedNewtonTiledCamera

        return FoveatedNewtonTiledCamera(cfg)
    if renderer_type in ("isaac_rtx", "ovrtx"):
        if not cfg.tiled:
            raise NotImplementedError("Non-tiled foveated RTX rendering is not implemented. Use tiled=True.")
        from .foveated_rtx_tiled_camera import FoveatedRTXTiledCamera

        return FoveatedRTXTiledCamera(cfg)
    raise ValueError(f"Unsupported foveated renderer {renderer_type!r}; select 'newton_warp', 'isaac_rtx', or 'ovrtx'.")


def __getattr__(name: str) -> type | Callable[..., wp.array]:
    if name == "FoveatedTiledCameraCfg":
        from .foveated_tiled_camera_cfg import FoveatedTiledCameraCfg

        return FoveatedTiledCameraCfg
    if name == "FoveatedNewtonTiledCamera":
        from .foveated_newton_tiled_camera import FoveatedNewtonTiledCamera

        return FoveatedNewtonTiledCamera
    if name == "FoveatedRTXTiledCamera":
        from .foveated_rtx_tiled_camera import FoveatedRTXTiledCamera

        return FoveatedRTXTiledCamera
    if name == "compute_foveated_ray_array":
        from .foveated_rays import compute_foveated_ray_array

        return compute_foveated_ray_array
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
