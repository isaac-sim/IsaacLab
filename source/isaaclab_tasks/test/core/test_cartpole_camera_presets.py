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

from isaaclab.utils import validate

from isaaclab_tasks.core.cartpole.cartpole_direct_camera_env_cfg import (
    CartpoleCameraEnvCfg as CartpoleDirectCameraEnvCfg,
)
from isaaclab_tasks.core.cartpole.cartpole_manager_camera_env_cfg import (
    CartpoleCameraEnvCfg as CartpoleManagerCameraEnvCfg,
)
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

_ENV_CFG_CLASSES = [
    pytest.param(CartpoleDirectCameraEnvCfg, id="direct"),
    pytest.param(CartpoleManagerCameraEnvCfg, id="manager"),
]


@pytest.mark.parametrize("cfg_cls", _ENV_CFG_CLASSES)
def test_newton_renderer_rejects_simple_shading(cfg_cls):
    """The Newton Warp renderer cannot shade, so this combination must not resolve."""
    cfg = resolve_presets(cfg_cls(), {"newton_warp", "simple_shading_full_mdl"})
    with pytest.raises(ValueError, match="simple_shading_full_mdl"):
        validate(cfg)


@pytest.mark.parametrize("cfg_cls", _ENV_CFG_CLASSES)
@pytest.mark.parametrize("renderer_preset", ["isaacsim_rtx", "ovrtx"])
def test_rtx_renderers_accept_simple_shading(cfg_cls, renderer_preset):
    """The RTX backends render every data type and must stay unaffected by the guard."""
    cfg = resolve_presets(cfg_cls(), {renderer_preset, "simple_shading_full_mdl"})
    validate(cfg)  # must not raise


@pytest.mark.parametrize("cfg_cls", _ENV_CFG_CLASSES)
def test_newton_renderer_accepts_supported_data_types(cfg_cls):
    """The guard must not reject data types the Newton Warp renderer does publish."""
    cfg = resolve_presets(cfg_cls(), {"newton_warp", "rgb"})
    validate(cfg)  # must not raise


def test_foveated_cartpole_presets_connect_the_camera_and_stock_ppo():
    """The registered task selects matching image data and runs without teacher-specific inputs or training code."""
    pytest.importorskip("isaaclab_fovi")
    pytest.importorskip("fovi")
    task = "IsaacContrib-Cartpole-Camera-Fovi"
    for data_type in ("rgb", "depth"):
        raw_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
        cfg = resolve_presets(raw_cfg, {data_type, "newton_warp"})
        validate(cfg)
        image = cfg.observations.policy.image.params
        assert image["data_type"] == data_type
        assert cfg.scene.tiled_camera.data_types == [data_type]
        assert image["sensor_cfg"].name == cfg.actions.gaze.asset_name
        assert not hasattr(cfg.observations, "oracle")

    agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    assert agent.obs_groups == {"actor": ["policy", "gaze"], "critic": ["critic", "gaze"]}
    assert agent.algorithm.class_name == "PPO"
    assert agent.actor.distribution_cfg.class_name == "GaussianDistribution"
