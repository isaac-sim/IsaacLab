# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the cartpole camera environment preset combinations.

The Newton Warp renderer cannot produce the ``simple_shading_*`` data types, but the
cartpole camera tasks offer both as independent ``presets=`` selectors. These tests resolve
the real configs and check that the incompatible combinations are rejected at config
resolution time, while the RTX backends keep accepting every data type. No simulator is
required: preset resolution and ``validate()`` are pure config operations.
"""

import types

import pytest
from isaaclab_newton.renderers import NewtonWarpRenderer, NewtonWarpRendererCfg

from isaaclab_tasks.core.cartpole.cartpole_direct_camera_env_cfg import (
    CartpoleCameraEnvCfg as CartpoleDirectCameraEnvCfg,
)
from isaaclab_tasks.core.cartpole.cartpole_manager_camera_env_cfg import (
    CartpoleCameraEnvCfg as CartpoleManagerCameraEnvCfg,
)
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.presets import NEWTON_WARP_SUPPORTED_DATA_TYPES

_ENV_CFG_CLASSES = [
    pytest.param(CartpoleDirectCameraEnvCfg, id="direct"),
    pytest.param(CartpoleManagerCameraEnvCfg, id="manager"),
]

_SIMPLE_SHADING_PRESETS = [
    "simple_shading_constant_diffuse",
    "simple_shading_diffuse_mdl",
    "simple_shading_full_mdl",
]

_RTX_RENDERER_PRESETS = ["isaacsim_rtx", "ovrtx"]


def _resolve(cfg_cls, presets):
    """Resolve a cartpole camera env config for the given preset selection."""
    return resolve_presets(cfg_cls(), set(presets))


@pytest.mark.parametrize("cfg_cls", _ENV_CFG_CLASSES)
@pytest.mark.parametrize("data_type_preset", _SIMPLE_SHADING_PRESETS)
def test_newton_renderer_rejects_simple_shading(cfg_cls, data_type_preset):
    """The Newton Warp renderer cannot shade, so these combinations must not resolve."""
    cfg = _resolve(cfg_cls, ["newton_renderer", data_type_preset])
    with pytest.raises(ValueError, match=data_type_preset):
        cfg.validate()


@pytest.mark.parametrize("cfg_cls", _ENV_CFG_CLASSES)
@pytest.mark.parametrize("data_type_preset", _SIMPLE_SHADING_PRESETS)
@pytest.mark.parametrize("renderer_preset", _RTX_RENDERER_PRESETS)
def test_rtx_renderers_accept_simple_shading(cfg_cls, data_type_preset, renderer_preset):
    """The RTX backends render every data type and must stay unaffected by the guard."""
    cfg = _resolve(cfg_cls, [renderer_preset, data_type_preset])
    cfg.validate()  # must not raise


@pytest.mark.parametrize("cfg_cls", _ENV_CFG_CLASSES)
@pytest.mark.parametrize("data_type_preset", ["rgb", "depth", "albedo", "semantic_segmentation"])
def test_newton_renderer_accepts_supported_data_types(cfg_cls, data_type_preset):
    """Data types the Newton Warp renderer publishes must keep resolving."""
    cfg = _resolve(cfg_cls, ["newton_renderer", data_type_preset])
    cfg.validate()  # must not raise


def test_supported_data_types_match_renderer_contract():
    """The guard's allow-list must not drift from what the renderer publishes."""
    # ``supported_output_types`` only reads ``self.cfg``; the renderer itself needs a running
    # simulation to construct, so call it against a config-only stand-in.
    stub = types.SimpleNamespace(cfg=NewtonWarpRendererCfg())
    published = {str(kind) for kind in NewtonWarpRenderer.supported_output_types(stub)}
    assert published == set(NEWTON_WARP_SUPPORTED_DATA_TYPES)
