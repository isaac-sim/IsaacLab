# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Renderer/data-type preset compatibility for the cartpole camera tasks.

The Newton Warp renderer cannot produce the ``simple_shading_*`` data types, which the tasks
offer as an independent ``presets=`` selector. Preset resolution and ``validate()`` are pure
config operations, so no simulator is launched here.
"""

import pytest

from isaaclab_tasks.core.cartpole.cartpole_direct_camera_env_cfg import (
    CartpoleCameraEnvCfg as CartpoleDirectCameraEnvCfg,
)
from isaaclab_tasks.core.cartpole.cartpole_manager_camera_env_cfg import (
    CartpoleCameraEnvCfg as CartpoleManagerCameraEnvCfg,
)
from isaaclab_tasks.utils.hydra import resolve_presets

_ENV_CFG_CLASSES = [
    pytest.param(CartpoleDirectCameraEnvCfg, id="direct"),
    pytest.param(CartpoleManagerCameraEnvCfg, id="manager"),
]


@pytest.mark.parametrize("cfg_cls", _ENV_CFG_CLASSES)
def test_newton_renderer_rejects_simple_shading(cfg_cls):
    """The Newton Warp renderer cannot shade, so this combination must not resolve."""
    cfg = resolve_presets(cfg_cls(), {"newton_renderer", "simple_shading_full_mdl"})
    with pytest.raises(ValueError, match="simple_shading_full_mdl"):
        cfg.validate()


@pytest.mark.parametrize("cfg_cls", _ENV_CFG_CLASSES)
@pytest.mark.parametrize("renderer_preset", ["isaacsim_rtx", "ovrtx"])
def test_rtx_renderers_accept_simple_shading(cfg_cls, renderer_preset):
    """The RTX backends render every data type and must stay unaffected by the guard."""
    cfg = resolve_presets(cfg_cls(), {renderer_preset, "simple_shading_full_mdl"})
    cfg.validate()  # must not raise


@pytest.mark.parametrize("cfg_cls", _ENV_CFG_CLASSES)
def test_newton_renderer_accepts_supported_data_types(cfg_cls):
    """The guard must not reject data types the Newton Warp renderer does publish."""
    cfg = resolve_presets(cfg_cls(), {"newton_renderer", "rgb"})
    cfg.validate()  # must not raise
