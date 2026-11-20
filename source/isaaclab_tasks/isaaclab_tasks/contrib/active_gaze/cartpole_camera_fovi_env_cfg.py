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
from isaaclab_fovi.sensors import FoveatedCameraCfg

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

from .active_gaze_cfg import GazeObservationsCfg, OracleObservationsCfg

POLE_TIP = (0.0, 0.0, 1.0)
"""The pole's tip in the pole's frame [m]."""

RETINA_RES = 32
"""Retina samples across; the stock camera is 96 pixels across."""

_STOCK_CAMERA = CartpoleTiledCameraCfg().rgb
_RGB_OBSERVATIONS = image_observations_cfg("rgb")
_RGB_CAMERA = FoveatedCameraCfg(
    prim_path=_STOCK_CAMERA.prim_path,
    offset=_STOCK_CAMERA.offset,
    spawn=_STOCK_CAMERA.spawn,
    renderer_cfg=_STOCK_CAMERA.renderer_cfg,
    data_types=["rgb"],
    foveated_res=RETINA_RES,
    mask_to_sensor_fov=True,
)


@configclass
class ObservationsCfg:
    policy: ObservationGroupCfg = _RGB_OBSERVATIONS.policy
    critic: ObservationGroupCfg = _RGB_OBSERVATIONS.critic
    gaze: ObservationGroupCfg = GazeObservationsCfg()
    oracle: ObservationGroupCfg = OracleObservationsCfg(
        labels=ObservationTermCfg(
            func=mdp.gaze_teacher_labels,
            params={"target_cfg": SceneEntityCfg("robot", body_names="pole"), "offset": POLE_TIP},
        )
    )


@configclass
class SceneCfg(CartpoleCameraSceneCfg):
    tiled_camera = preset(
        default=_RGB_CAMERA,
        depth=_RGB_CAMERA.replace(data_types=["depth"]),
        rgb=_RGB_CAMERA,
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

    rgb = BaseCfg(observations=ObservationsCfg())
    depth = BaseCfg(observations=ObservationsCfg(policy=image_observations_cfg("depth").policy))
    default = rgb
