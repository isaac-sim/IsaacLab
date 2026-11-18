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
    assert cfg.scene.lidar.frame_of_reference == "SENSOR"
    assert cfg.scene.lidar.far_range == 12.0
    assert cfg.scene.lidar.renderer_cfg.motion_bvh == "auto"
    assert cfg.scene.tiled_camera.renderer_cfg.motion_bvh == "auto"
    assert cfg.scene.lidar.renderer_cfg == cfg.scene.tiled_camera.renderer_cfg
    if cfg_cls is CartpoleDirectCameraEnvCfg:
        from isaaclab_tasks.core.cartpole.cartpole_common import (
            CARTPOLE_LIDAR_AZIMUTH_COUNT,
            CARTPOLE_LIDAR_ELEVATION_COUNT,
        )

        assert cfg.observation_space == [1, CARTPOLE_LIDAR_ELEVATION_COUNT, CARTPOLE_LIDAR_AZIMUTH_COUNT]
        assert cfg.state_space == 4


def test_lidar_preset_selects_cnn_agents():
    """The lidar observation is a range image, so the registered agents use the camera CNN."""
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
    assert "cnn" in rl_games_cfg["params"]["network"]
    assert rl_games_config["name"] == "cartpole_lidar"
    assert rl_games_config["normalize_input"] is False

    rsl_rl_cfg = resolve_presets(CartpoleCameraPPORunnerPresetsCfg(), {"lidar"})
    assert rsl_rl_cfg.actor.class_name == "CNNModel"
    assert rsl_rl_cfg.actor.obs_normalization is False
    assert rsl_rl_cfg.obs_groups == {"actor": ["policy"], "critic": ["critic"]}

    direct_rl_games_cfg = resolve_presets(rl_games_direct_camera_cfg(), {"lidar"})
    direct_rl_games_config = direct_rl_games_cfg["params"]["config"]
    assert "cnn" in direct_rl_games_cfg["params"]["network"]
    assert direct_rl_games_config["name"] == "cartpole_lidar_direct"
    assert direct_rl_games_config["normalize_input"] is False
    default_rl_games_cfg = resolve_presets(rl_games_direct_camera_cfg(), set())
    assert "cnn" in default_rl_games_cfg["params"]["network"]

    direct_rsl_rl_cfg = resolve_presets(CartpoleCameraDirectPPORunnerPresetsCfg(), {"lidar"})
    assert direct_rsl_rl_cfg.actor.class_name == "CNNModel"
    assert direct_rsl_rl_cfg.actor.obs_normalization is False
    assert direct_rsl_rl_cfg.obs_groups == {"actor": ["policy"], "critic": ["critic"]}
    assert direct_rsl_rl_cfg.experiment_name == "cartpole_lidar_direct"
    default_rsl_rl_cfg = resolve_presets(CartpoleCameraDirectPPORunnerPresetsCfg(), set())
    assert default_rsl_rl_cfg.actor.class_name == "CNNModel"

    direct_skrl_cfg = resolve_presets(skrl_direct_camera_cfg(), {"lidar"})
    assert "conv2d" in direct_skrl_cfg["models"]["policy"]["network"][0]["layers"][0]
    assert "conv2d" in direct_skrl_cfg["models"]["value"]["network"][0]["layers"][0]
    assert direct_skrl_cfg["agent"]["state_preprocessor"] is None
    assert direct_skrl_cfg["agent"]["experiment"]["directory"] == "cartpole_lidar_direct"
    default_skrl_cfg = resolve_presets(skrl_direct_camera_cfg(), set())
    assert "conv2d" in default_skrl_cfg["models"]["policy"]["network"][0]["layers"][0]


def test_lidar_range_image_places_rays_on_the_elevation_azimuth_grid():
    """A hit on the first ray is the top-left pixel, and a miss stays zero."""
    import torch

    from isaaclab_tasks.core.cartpole.cartpole_common import (
        CARTPOLE_LIDAR_AZIMUTH_COUNT,
        CARTPOLE_LIDAR_ELEVATION_COUNT,
        CARTPOLE_LIDAR_FAR_RANGE,
        cartpole_lidar_range_image,
    )

    num_rays = CARTPOLE_LIDAR_ELEVATION_COUNT * CARTPOLE_LIDAR_AZIMUTH_COUNT
    points = torch.full((2, num_rays, 3), float("nan"))
    points[:, 0] = torch.tensor([6.0, 0.0, 0.0])
    image = cartpole_lidar_range_image(points)

    assert image.shape == (2, CARTPOLE_LIDAR_ELEVATION_COUNT, CARTPOLE_LIDAR_AZIMUTH_COUNT)
    assert image[0, 0, 0] == pytest.approx(6.0 / CARTPOLE_LIDAR_FAR_RANGE)
    assert image[0, 0, 1] == 0.0
    assert image[0, 1, 0] == 0.0
    assert float(image.max()) <= 1.0


@pytest.mark.parametrize("cfg_cls", _ENV_CFG_CLASSES)
def test_rgb_preset_has_no_lidar(cfg_cls):
    """Existing camera presets do not gain a lidar or change renderer."""
    cfg = resolve_presets(cfg_cls(), {"rgb"})
    validate(cfg)
    assert not hasattr(cfg.scene, "lidar")
    assert cfg.scene.tiled_camera.data_types == ["rgb"]
