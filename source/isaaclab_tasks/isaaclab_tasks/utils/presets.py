# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

from isaaclab.renderers.renderer_cfg import RendererCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

NEWTON_WARP_SUPPORTED_DATA_TYPES: frozenset[str] = frozenset(
    {
        "rgb",
        "rgba",
        "rgb_hdr",
        "albedo",
        "depth",
        "distance_to_camera",
        "distance_to_image_plane",
        "normals",
        "semantic_segmentation",
        "instance_segmentation",
    }
)
"""Camera data types the Newton Warp renderer can produce.

Mirrors the keys published by
:meth:`~isaaclab_newton.renderers.NewtonWarpRenderer.supported_output_types`. The renderer
raises on unsupported types only once the camera allocates its buffers, i.e. after the
simulation has started, so configs use this set to fail at config-resolution time instead.
"""


@configclass
class _AutoRtxRendererCfg(RendererCfg):
    renderer_type: str = "auto_rtx"


@configclass
class MultiBackendRendererCfg(PresetCfg):
    rtx: _AutoRtxRendererCfg = _AutoRtxRendererCfg()
    ovrtx: OVRTXRendererCfg = OVRTXRendererCfg()
    isaacsim_rtx: IsaacRtxRendererCfg = IsaacRtxRendererCfg()
    newton_renderer: NewtonWarpRendererCfg = NewtonWarpRendererCfg()
    default: NewtonWarpRendererCfg = NewtonWarpRendererCfg()


def validate_warp_renderer_data_types(camera_cfg: CameraCfg, camera_name: str) -> None:
    """Reject camera data types the Newton Warp renderer cannot render.

    No-op unless the camera resolved to the Newton Warp renderer; the RTX backends accept
    every data type. Call this from a config's ``validate_config`` hook so an unsupported
    ``presets=`` combination fails during config resolution rather than at env construction.

    Args:
        camera_cfg: Concrete camera config, after preset resolution.
        camera_name: Camera attribute name, used in the error message.

    Raises:
        ValueError: If the camera requests data types the Newton Warp renderer does not publish.
    """
    renderer_type = getattr(getattr(camera_cfg, "renderer_cfg", None), "renderer_type", None)
    if renderer_type != "newton_warp":
        return
    unsupported = set(camera_cfg.data_types) - NEWTON_WARP_SUPPORTED_DATA_TYPES
    if unsupported:
        raise ValueError(
            f"Warp renderer only supports data types {sorted(NEWTON_WARP_SUPPORTED_DATA_TYPES)}, "
            f"but '{camera_name}' is configured with unsupported types: {sorted(unsupported)}. "
            "Choose a compatible preset, e.g. presets=newton_renderer,rgb."
        )


def set_isaac_rtx_global_settings(renderer_cfg: Any, **settings: Any) -> None:
    """Set Isaac RTX settings on direct or preset-wrapped renderer configs."""
    visited: set[int] = set()

    def _visit(cfg: Any) -> None:
        if cfg is None or id(cfg) in visited:
            return
        visited.add(id(cfg))
        if getattr(cfg, "renderer_type", None) == "isaac_rtx" and hasattr(cfg, "global_settings"):
            for key, value in settings.items():
                setattr(cfg.global_settings, key, value)
        for attr_name in ("default", "isaacsim_rtx"):
            _visit(getattr(cfg, attr_name, None))

    _visit(renderer_cfg)
