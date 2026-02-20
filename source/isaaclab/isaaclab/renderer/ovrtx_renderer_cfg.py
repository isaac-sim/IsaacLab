# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for OVRTX Renderer."""

from isaaclab.utils import configclass

from .renderer_cfg import RendererCfg


@configclass
class OVRTXRendererCfg(RendererCfg):
    """Configuration for OVRTX Renderer.
    
    The OVRTX renderer uses the ovrtx library for high-fidelity RTX-based rendering.
    """

    renderer_type: str = "ov_rtx"
    """Type identifier for OVRTX renderer."""
    
    simple_shading_mode: bool = True
    """Whether to use simple shading mode (default: True).
    
    When enabled, this mode:
    - Uses SimpleShadingSD RenderVar instead of LdrColor for RGB rendering
    
    This provides faster, simpler rendering suitable for many vision-based tasks.
    Set to False to use full RTX path-traced rendering with LdrColor.
    """
    
    image_folder: str | None = None
    """Optional output directory for saving rendered images.
    
    When set, all rendered images (RGB, depth, albedo, semantic segmentation) will be
    saved to this directory. If None, no images will be saved to disk.
    Default: None (no image saving).
    """