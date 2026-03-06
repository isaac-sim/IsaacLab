# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVRTX renderer backend tests: OVRTXRendererCfg yields OVRTXRenderer."""

from isaaclab.renderers import Renderer
from isaaclab.renderers.base_renderer import BaseRenderer

from isaaclab_ov.renderers import OVRTXRendererCfg, OVRTXRenderer


def test_ovrtx_renderer_cfg_yields_ovrtx_renderer():
    """OVRTXRendererCfg -> Renderer(cfg) returns OVRTXRenderer."""
    cfg = OVRTXRendererCfg()
    renderer = Renderer(cfg)
    assert type(renderer) is OVRTXRenderer
    assert isinstance(renderer, BaseRenderer)
