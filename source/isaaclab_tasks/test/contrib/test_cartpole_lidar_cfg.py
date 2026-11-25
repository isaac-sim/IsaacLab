# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config contracts for the Cartpole lidar tasks. No simulator is launched."""

import pytest
from isaaclab_ov.renderers import OVRTXRendererCfg

from isaaclab.sensors import LidarCfg
from isaaclab.utils import validate

from isaaclab_tasks.contrib.cartpole_lidar.cartpole_lidar_common import (
    CARTPOLE_LIDAR_AZIMUTH_COUNT,
    CARTPOLE_LIDAR_ELEVATION_COUNT,
    CARTPOLE_LIDAR_FAR_RANGE,
    CARTPOLE_LIDAR_SUBSAMPLING,
    cartpole_lidar_range_image,
)
from isaaclab_tasks.contrib.cartpole_lidar.cartpole_lidar_direct_env_cfg import (
    CartpoleLidarEnvCfg as CartpoleLidarDirectEnvCfg,
)
from isaaclab_tasks.contrib.cartpole_lidar.cartpole_lidar_env_cfg import (
    CartpoleLidarEnvCfg as CartpoleLidarManagerEnvCfg,
)
from isaaclab_tasks.utils.hydra import resolve_presets, resolve_task_config

pytestmark = pytest.mark.kitless

_ENV_CFGS = [
    pytest.param(CartpoleLidarManagerEnvCfg, id="manager"),
    pytest.param(CartpoleLidarDirectEnvCfg, id="direct"),
]


@pytest.mark.parametrize("cfg_cls", _ENV_CFGS)
def test_lidar_task_uses_ovrtx_and_drops_the_camera(cfg_cls):
    """The lidar scene pins OVRTX with motion BVH and has no tiled camera."""
    cfg = resolve_presets(cfg_cls())
    validate(cfg)
    assert isinstance(cfg.scene.lidar, LidarCfg)
    assert isinstance(cfg.scene.lidar.renderer_cfg, OVRTXRendererCfg)
    assert getattr(cfg.scene, "tiled_camera", None) is None
    assert cfg.scene.lidar.renderer_cfg.motion_bvh == "auto"
    assert cfg.scene.num_envs == 64
    if cfg_cls is CartpoleLidarDirectEnvCfg:
        assert cfg.observation_space == [1, CARTPOLE_LIDAR_ELEVATION_COUNT, CARTPOLE_LIDAR_AZIMUTH_COUNT]
        assert cfg.state_space == 4


def test_lidar_tasks_select_cnn_agents():
    """The lidar observation is a range image, so the registered tasks select the camera CNN."""
    import isaaclab_tasks  # noqa: F401

    _, rl_games_cfg = resolve_task_config("IsaacContrib-Cartpole-Lidar", "rl_games_cfg_entry_point", overrides=[])
    rl_games_config = rl_games_cfg["params"]["config"]
    assert "cnn" in rl_games_cfg["params"]["network"]
    assert rl_games_config["name"] == "cartpole_camera"
    assert rl_games_config["normalize_input"] is False

    _, rsl_rl_cfg = resolve_task_config("IsaacContrib-Cartpole-Lidar", "rsl_rl_cfg_entry_point", overrides=[])
    assert rsl_rl_cfg.actor.class_name == "CNNModel"
    assert rsl_rl_cfg.actor.obs_normalization is False
    assert rsl_rl_cfg.obs_groups == {"actor": ["policy"], "critic": ["critic"]}

    _, direct_rl_games_cfg = resolve_task_config(
        "IsaacContrib-Cartpole-Lidar-Direct", "rl_games_cfg_entry_point", overrides=[]
    )
    direct_rl_games_config = direct_rl_games_cfg["params"]["config"]
    assert "cnn" in direct_rl_games_cfg["params"]["network"]
    assert direct_rl_games_config["name"] == "cartpole_camera"
    assert direct_rl_games_config["normalize_input"] is False

    _, direct_rsl_rl_cfg = resolve_task_config(
        "IsaacContrib-Cartpole-Lidar-Direct", "rsl_rl_cfg_entry_point", overrides=[]
    )
    assert direct_rsl_rl_cfg.actor.class_name == "CNNModel"
    assert direct_rsl_rl_cfg.actor.obs_normalization is False
    assert direct_rsl_rl_cfg.obs_groups == {"actor": ["policy"], "critic": ["critic"]}
    assert direct_rsl_rl_cfg.experiment_name == "cartpole_camera_direct"


def test_lidar_range_image_places_rays_on_the_elevation_azimuth_grid():
    """An interior ray lands on its elevation row and azimuth column, and a miss stays zero."""
    import torch

    elevation, azimuth = 3, 5
    ray = elevation * CARTPOLE_LIDAR_AZIMUTH_COUNT + azimuth
    num_rays = CARTPOLE_LIDAR_ELEVATION_COUNT * CARTPOLE_LIDAR_AZIMUTH_COUNT
    points = torch.full((2, num_rays, 3), float("nan"))
    points[:, ray] = torch.tensor([6.0, 0.0, 0.0])
    image = cartpole_lidar_range_image(points)

    assert image.shape == (2, CARTPOLE_LIDAR_ELEVATION_COUNT, CARTPOLE_LIDAR_AZIMUTH_COUNT)
    assert image[0, elevation, azimuth] == pytest.approx(6.0 / CARTPOLE_LIDAR_FAR_RANGE)
    assert image[0, azimuth, elevation] == 0.0
    assert image[0, 0, 0] == 0.0
    assert float(image.max()) <= 1.0

    published = num_rays // CARTPOLE_LIDAR_SUBSAMPLING
    sparse = torch.full((2, published, 3), float("nan"))
    sparse[:, ray // CARTPOLE_LIDAR_SUBSAMPLING] = torch.tensor([6.0, 0.0, 0.0])
    sparse_image = cartpole_lidar_range_image(sparse)
    assert sparse_image.shape == image.shape
    assert sparse_image[0, elevation, azimuth] == pytest.approx(6.0 / CARTPOLE_LIDAR_FAR_RANGE)
    assert sparse_image[0, 0, 1] == 0.0
