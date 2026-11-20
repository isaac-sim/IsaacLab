# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cartpole with a foveated camera, in RGB or depth.

The stock camera task with the camera foveated and two gaze actions. The teacher looks at the pole's tip. The retina
is :data:`RETINA_RES` samples across, enough for a task that only needs to see a bar. The policy gets two gaze actions,
whose noise is fixed, and the teacher's gaze labels supervise them during training.
"""

from isaaclab_fovi import mdp

from isaaclab.managers import ObservationGroupCfg, SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.core.cartpole.cartpole_common import CartpoleTiledCameraCfg
from isaaclab_tasks.core.cartpole.cartpole_manager_camera_env_cfg import (
    CartpoleCameraEnvCfg,
    CartpoleCameraSceneCfg,
    image_observations_cfg,
)
from isaaclab_tasks.core.cartpole.cartpole_manager_env_cfg import ActionsCfg as CartpoleActionsCfg
from isaaclab_tasks.utils import PresetCfg

from .active_gaze_cfg import GazeObservationsCfg, foveated_presets, oracle_observations, select_presets

POLE_TIP = (0.0, 0.0, 1.0)
"""The pole's tip in the pole's frame [m]."""

RETINA_RES = 32
"""Retina samples across; the stock camera is 96 pixels across."""


def observations(data_type: str) -> object:
    """The stock camera observations of ``data_type``, the gaze, and the teacher's labels."""
    camera = image_observations_cfg(data_type)

    @configclass
    class ObservationsCfg:
        policy: ObservationGroupCfg = camera.policy
        critic: ObservationGroupCfg = camera.critic
        gaze: ObservationGroupCfg = GazeObservationsCfg()
        oracle: ObservationGroupCfg = oracle_observations(SceneEntityCfg("robot", body_names="pole"), POLE_TIP)

    return ObservationsCfg()


@configclass
class SceneCfg(CartpoleCameraSceneCfg):
    tiled_camera = select_presets(foveated_presets(CartpoleTiledCameraCfg(), RETINA_RES), ("rgb", "depth"))


@configclass
class ActionsCfg(CartpoleActionsCfg):
    # The gaze must be the last action term.
    gaze = mdp.GazeActionCfg(asset_name="tiled_camera")


@configclass
class CartpoleCameraFoviEnvCfg(PresetCfg):
    """Cartpole with a foveated camera; ``presets=depth`` selects depth images."""

    @configclass
    class BaseCfg(CartpoleCameraEnvCfg.BaseCartpoleCameraEnvCfg):
        scene: SceneCfg = SceneCfg(num_envs=512, env_spacing=20.0)
        actions: ActionsCfg = ActionsCfg()

    rgb = BaseCfg(observations=observations("rgb"))
    depth = BaseCfg(observations=observations("depth"))
    default = rgb
