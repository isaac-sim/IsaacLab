# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cartpole with a foveated camera, in RGB or depth.

The stock camera task with the camera foveated and two gaze actions. The teacher looks at the pole's tip. The retina
is :data:`RETINA_RES` samples across, enough for a task that only needs to see a bar. The policy gets two gaze actions,
whose noise is fixed, and the teacher's gaze labels supervise them during training.
"""

from dataclasses import MISSING

from isaaclab_fovi import mdp

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.core.cartpole.cartpole_common import CartpoleTiledCameraCfg
from isaaclab_tasks.core.cartpole.cartpole_manager_camera_env_cfg import (
    CartpoleCameraEnvCfg,
    CartpoleCameraSceneCfg,
    image_observations_cfg,
)
from isaaclab_tasks.core.cartpole.cartpole_manager_env_cfg import ActionsCfg as CartpoleActionsCfg
from isaaclab_tasks.utils import PresetCfg, preset

from .active_gaze_cfg import GazeObservationsCfg, OracleObservationsCfg, foveated_camera

POLE_TIP = (0.0, 0.0, 1.0)
"""The pole's tip in the pole's frame [m]."""

RETINA_RES = 32
"""Retina samples across; the stock camera is 96 pixels across."""

_STOCK_CAMERAS = CartpoleTiledCameraCfg()


@configclass
class ObservationsCfg:
    policy: ObservationGroupCfg = MISSING
    critic: ObservationGroupCfg = MISSING
    gaze: ObservationGroupCfg = GazeObservationsCfg()
    oracle: ObservationGroupCfg = OracleObservationsCfg(
        labels=ObservationTermCfg(
            func=mdp.gaze_teacher_labels,
            params={"target_cfg": SceneEntityCfg("robot", body_names="pole"), "offset": POLE_TIP},
        )
    )


def _observations(data_type: str) -> ObservationsCfg:
    camera = image_observations_cfg(data_type)
    return ObservationsCfg(policy=camera.policy, critic=camera.critic)


@configclass
class SceneCfg(CartpoleCameraSceneCfg):
    tiled_camera = preset(
        default=foveated_camera(_STOCK_CAMERAS.default, RETINA_RES),
        depth=foveated_camera(_STOCK_CAMERAS.depth, RETINA_RES),
        rgb=foveated_camera(_STOCK_CAMERAS.rgb, RETINA_RES),
    )


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

    rgb = BaseCfg(observations=_observations("rgb"))
    depth = BaseCfg(observations=_observations("depth"))
    default = rgb
