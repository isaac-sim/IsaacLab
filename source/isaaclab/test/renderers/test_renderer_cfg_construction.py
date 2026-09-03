# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Architecture gates for declarative renderer construction."""

import pytest

import isaaclab.renderers as renderers
from isaaclab.renderers import RendererCfg
from isaaclab.utils.string import ResolvableString

pytestmark = [pytest.mark.integration, pytest.mark.rendering]


@pytest.mark.parametrize(
    "module_name,cfg_name,implementation",
    [
        ("isaaclab_physx.renderers", "IsaacRtxRendererCfg", "IsaacRtxRenderer"),
        ("isaaclab_newton.renderers", "NewtonWarpRendererCfg", "NewtonWarpRenderer"),
        ("isaaclab_ov.renderers", "OVRTXRendererCfg", "OVRTXRenderer"),
    ],
)
def test_renderer_cfg_names_its_implementation(module_name, cfg_name, implementation):
    """Every concrete renderer cfg resolves the class used by ``cfg.class_type(cfg)``."""
    cfg_type = getattr(pytest.importorskip(module_name), cfg_name)
    class_type = cfg_type().class_type
    assert isinstance(class_type, ResolvableString)
    assert class_type.__name__ == implementation


def test_renderer_construction_has_no_factory_api():
    assert not hasattr(renderers, "Renderer")
    assert not any(hasattr(RendererCfg, name) for name in ("build", "build_renderer", "clone_context"))
