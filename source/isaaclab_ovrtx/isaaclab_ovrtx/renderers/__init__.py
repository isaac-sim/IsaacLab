# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for OVRTX renderer backend."""

from .ovrtx_renderer import OVRTXRenderer
from .ovrtx_renderer_cfg import OVRTXRendererCfg

Renderer = OVRTXRenderer

__all__ = [
    "OVRTXRenderer",
    "OVRTXRendererCfg",
    "Renderer",
]
