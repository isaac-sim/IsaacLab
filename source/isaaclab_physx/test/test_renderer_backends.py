# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX renderer backend tests: IsaacRtxRendererCfg yields Isaac RTX renderer."""

import pytest

from isaaclab.renderers import Renderer
from isaaclab.renderers.base_renderer import BaseRenderer

from isaaclab_physx.renderers import IsaacRtxRendererCfg


def test_physx_renderer_factory_backend_mapping():
    """Renderer._get_backend(IsaacRtxRendererCfg()) == 'physx'."""
    cfg = IsaacRtxRendererCfg()
    assert Renderer._get_backend(cfg) == "physx"


def test_physx_renderer_cfg_yields_isaac_rtx_renderer():
    """IsaacRtxRendererCfg -> Renderer(cfg) returns IsaacRtxRenderer (requires omni/Isaac Sim)."""
    pytest.importorskip("omni.physics")
    from isaaclab_physx.renderers import IsaacRtxRenderer

    cfg = IsaacRtxRendererCfg()
    renderer = Renderer(cfg)
    assert type(renderer) is IsaacRtxRenderer
    assert isinstance(renderer, BaseRenderer)
