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
    
    use_ovrtx_cloning: bool = True
    """Whether to use OVRTX internal cloning for environments (default: True).
    
    When True (default):
    - Only the base environment (env_0) is exported to USD
    - OVRTX uses its efficient clone_usd() method to replicate environments
    - Significantly faster initialization for many environments (10-100x speedup)
    
    When False:
    - All environments are exported to the USD file (fully cloned stage)
    - OVRTX loads the complete scene without internal cloning
    - Slower but may be useful for debugging or special rendering requirements
    """