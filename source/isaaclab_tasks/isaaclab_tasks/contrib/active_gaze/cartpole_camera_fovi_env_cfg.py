# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera Cartpole with a foveated view and policy-controlled gaze."""

from dataclasses import MISSING

from isaaclab_fovi import mdp
from isaaclab_fovi.sensors import FoveatedCameraCfg

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.core.cartpole.cartpole_common import CartpoleTiledCameraCfg
from isaaclab_tasks.core.cartpole.cartpole_manager_camera_env_cfg import (
    CartpoleCameraEnvCfg,
    CartpoleCameraSceneCfg,
    image_observations_cfg,
)
from isaaclab_tasks.core.cartpole.cartpole_manager_env_cfg import ActionsCfg as CartpoleActionsCfg
from isaaclab_tasks.utils import PresetCfg


def _foveated_camera(camera: CameraCfg) -> FoveatedCameraCfg:
    """Apply the retina to the stock Cartpole camera's mount, optics, and renderer."""
    return FoveatedCameraCfg(
        prim_path=camera.prim_path,
        offset=camera.offset,
        spawn=camera.spawn,
        data_types=camera.data_types,
        renderer_cfg=camera.renderer_cfg,
        foveated_res=32,
        mask_to_sensor_fov=True,
    )


@configclass
class CameraPresetsCfg(PresetCfg):
    """Select the foveated RGB or depth output."""

    rgb = _foveated_camera(CartpoleTiledCameraCfg().rgb)
    depth = _foveated_camera(CartpoleTiledCameraCfg().depth)
    default = rgb


@configclass
class SceneCfg(CartpoleCameraSceneCfg):
    tiled_camera: CameraPresetsCfg = CameraPresetsCfg()


@configclass
class ActionsCfg(CartpoleActionsCfg):
    gaze = mdp.GazeActionCfg(asset_name="tiled_camera")


@configclass
class ObservationsCfg:
    @configclass
    class GazeCfg(ObservationGroupCfg):
        state = ObservationTermCfg(func=mdp.gaze_state)

    policy: ObservationGroupCfg = MISSING
    critic: ObservationGroupCfg = MISSING
    gaze: GazeCfg = GazeCfg()


def _observations(data_type: str) -> ObservationsCfg:
    images = image_observations_cfg(data_type)
    return ObservationsCfg(policy=images.policy, critic=images.critic)


@configclass
class CartpoleCameraFoviEnvCfg(PresetCfg):
    """Cartpole with gaze control and RGB or depth retinal observations."""

    @configclass
    class BaseCfg(CartpoleCameraEnvCfg.BaseCartpoleCameraEnvCfg):
        scene: SceneCfg = SceneCfg(num_envs=512, env_spacing=20.0)
        actions: ActionsCfg = ActionsCfg()

    rgb = BaseCfg(observations=_observations("rgb"))
    depth = BaseCfg(observations=_observations("depth"))
    default = rgb
