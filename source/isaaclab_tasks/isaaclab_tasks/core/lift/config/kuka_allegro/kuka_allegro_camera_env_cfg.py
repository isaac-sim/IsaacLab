# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera (vision) variants of the KukaAllegro Lift and Reorient tasks.

Each task is exposed as a :class:`~isaaclab_tasks.utils.PresetCfg` whose ``single_camera`` /
``duo_camera`` variants add base (and wrist) cameras plus the matching image observations on top of
the state env config in :mod:`.kuka_allegro_env_cfg`. The camera data type / resolution and
renderer backend remain ``presets=`` selectable through the camera configs.
"""

from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.core.lift.config.kuka_allegro.camera_cfg import (
    BaseTiledCameraCfg,
    DuoCameraObservationsCfg,
    SingleCameraObservationsCfg,
    WristTiledCameraCfg,
)
from isaaclab_tasks.core.lift.config.kuka_allegro.kuka_allegro_env_cfg import (
    KukaAllegroLiftEnvCfg,
    KukaAllegroReorientEnvCfg,
    KukaAllegroSceneCfg,
)
from isaaclab_tasks.utils import PresetCfg

_SCENE_KWARGS = {"num_envs": 4096, "env_spacing": 3, "replicate_physics": True}


@configclass
class SingleCameraSceneCfg(KukaAllegroSceneCfg):
    """KukaAllegro scene with a single base-mounted camera."""

    base_camera: CameraCfg = BaseTiledCameraCfg()


@configclass
class DuoCameraSceneCfg(KukaAllegroSceneCfg):
    """KukaAllegro scene with base-mounted and wrist-mounted cameras."""

    base_camera: CameraCfg = BaseTiledCameraCfg()
    wrist_camera: CameraCfg = WristTiledCameraCfg()


def _camera_env(
    base_cls: type[KukaAllegroReorientEnvCfg], scene_cls: type[KukaAllegroSceneCfg], obs_cls: type
) -> KukaAllegroReorientEnvCfg:
    """Build a camera env config by swapping a camera scene and image observations onto a state env."""
    return base_cls(scene=scene_cls(**_SCENE_KWARGS), observations=obs_cls())


@configclass
class KukaAllegroReorientCameraEnvCfg(PresetCfg):
    """Camera variants of the Kuka-Allegro reorientation environment."""

    single_camera = _camera_env(KukaAllegroReorientEnvCfg, SingleCameraSceneCfg, SingleCameraObservationsCfg)
    duo_camera = _camera_env(KukaAllegroReorientEnvCfg, DuoCameraSceneCfg, DuoCameraObservationsCfg)
    default = single_camera


@configclass
class KukaAllegroLiftCameraEnvCfg(PresetCfg):
    """Camera variants of the Kuka-Allegro lifting environment."""

    single_camera = _camera_env(KukaAllegroLiftEnvCfg, SingleCameraSceneCfg, SingleCameraObservationsCfg)
    duo_camera = _camera_env(KukaAllegroLiftEnvCfg, DuoCameraSceneCfg, DuoCameraObservationsCfg)
    default = single_camera
