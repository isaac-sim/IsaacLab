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


@pytest.mark.parametrize("cfg_cls", _ENV_CFG_CLASSES)
def test_lidar_preset_uses_one_ovrtx_renderer(cfg_cls):
    """The lidar preset pins the camera and the lidar to OVRTX with motion BVH enabled."""
    from isaaclab_ov.renderers import OVRTXRendererCfg

    from isaaclab.sensors import LidarCfg

    cfg = resolve_presets(cfg_cls(), {"lidar"})
    validate(cfg)
    assert isinstance(cfg.scene.lidar, LidarCfg)
    assert isinstance(cfg.scene.lidar.renderer_cfg, OVRTXRendererCfg)
    assert isinstance(cfg.scene.tiled_camera.renderer_cfg, OVRTXRendererCfg)
    assert cfg.scene.lidar.renderer_cfg.motion_bvh == "auto"
    assert cfg.scene.tiled_camera.renderer_cfg.motion_bvh == "auto"
    assert cfg.scene.lidar.renderer_cfg == cfg.scene.tiled_camera.renderer_cfg
    if cfg_cls is CartpoleDirectCameraEnvCfg:
        assert cfg.observation_space == 4 + len(cfg.scene.lidar.azimuth_deg) * 3
        assert cfg.state_space == 4


def test_lidar_preset_selects_mlp_agents():
    """The lidar observation is a vector, so the registered agents must stay off the image CNN."""
    from isaaclab_tasks.core.cartpole.agents import (
        rl_games_camera_cfg,
        rl_games_direct_camera_cfg,
        skrl_direct_camera_cfg,
    )
    from isaaclab_tasks.core.cartpole.agents.rsl_rl_ppo_cfg import (
        CartpoleCameraDirectPPORunnerPresetsCfg,
        CartpoleCameraPPORunnerPresetsCfg,
    )

    rl_games_cfg = resolve_presets(rl_games_camera_cfg(), {"lidar"})
    rl_games_config = rl_games_cfg["params"]["config"]
    assert "cnn" not in rl_games_cfg["params"]["network"]
    assert rl_games_config["name"] == "cartpole_lidar"
    assert "minibatch_size" not in rl_games_config
    assert rl_games_config["horizon_length"] % rl_games_config["minibatch_size_per_env"] == 0

    rsl_rl_cfg = resolve_presets(CartpoleCameraPPORunnerPresetsCfg(), {"lidar"})
    assert rsl_rl_cfg.actor.class_name == "MLPModel"
    assert rsl_rl_cfg.obs_groups == {"actor": ["policy"], "critic": ["critic"]}

    direct_rl_games_cfg = resolve_presets(rl_games_direct_camera_cfg(), {"lidar"})
    direct_rl_games_config = direct_rl_games_cfg["params"]["config"]
    assert "cnn" not in direct_rl_games_cfg["params"]["network"]
    assert direct_rl_games_config["name"] == "cartpole_lidar_direct"
    assert "minibatch_size" not in direct_rl_games_config
    assert direct_rl_games_config["horizon_length"] % direct_rl_games_config["minibatch_size_per_env"] == 0
    default_rl_games_cfg = resolve_presets(rl_games_direct_camera_cfg(), set())
    assert "cnn" in default_rl_games_cfg["params"]["network"]

    direct_rsl_rl_cfg = resolve_presets(CartpoleCameraDirectPPORunnerPresetsCfg(), {"lidar"})
    assert direct_rsl_rl_cfg.actor.class_name == "MLPModel"
    assert direct_rsl_rl_cfg.obs_groups == {"actor": ["policy"], "critic": ["critic"]}
    assert direct_rsl_rl_cfg.experiment_name == "cartpole_lidar_direct"
    default_rsl_rl_cfg = resolve_presets(CartpoleCameraDirectPPORunnerPresetsCfg(), set())
    assert default_rsl_rl_cfg.actor.class_name == "CNNModel"

    direct_skrl_cfg = resolve_presets(skrl_direct_camera_cfg(), {"lidar"})
    assert direct_skrl_cfg["models"]["policy"]["network"][0]["layers"] == [256, 128]
    assert direct_skrl_cfg["models"]["value"]["network"][0]["layers"] == [256, 128]
    assert direct_skrl_cfg["agent"]["state_preprocessor"] == "RunningStandardScaler"
    assert direct_skrl_cfg["agent"]["experiment"]["directory"] == "cartpole_lidar_direct"
    default_skrl_cfg = resolve_presets(skrl_direct_camera_cfg(), set())
    assert "conv2d" in default_skrl_cfg["models"]["policy"]["network"][0]["layers"][0]


@pytest.mark.parametrize("cfg_cls", _ENV_CFG_CLASSES)
def test_rgb_preset_has_no_lidar(cfg_cls):
    """Existing camera presets do not gain a lidar or change renderer."""
    cfg = resolve_presets(cfg_cls(), {"rgb"})
    validate(cfg)
    assert not hasattr(cfg.scene, "lidar")
    assert cfg.scene.tiled_camera.data_types == ["rgb"]
