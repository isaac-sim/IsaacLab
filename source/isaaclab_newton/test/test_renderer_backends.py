# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton renderer backend tests: NewtonWarpRendererCfg backend mapping."""

from isaaclab.renderers import Renderer
from isaaclab_newton.renderers import NewtonWarpRendererCfg


def test_newton_renderer_factory_backend_mapping():
    """Renderer._get_backend(NewtonWarpRendererCfg()) == 'newton' (renderer backend)."""
    cfg = NewtonWarpRendererCfg()
    assert Renderer._get_backend(cfg) == "newton"


def test_newton_renderer_cfg_type():
    """NewtonWarpRendererCfg has expected renderer_type for factory."""
    cfg = NewtonWarpRendererCfg()
    assert getattr(cfg, "renderer_type", None) == "newton_warp"
