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

from dataclasses import dataclass
from typing import Any

from isaaclab.sensors import CameraCfg
from isaaclab.utils import config_field

from isaaclab_tasks.utils import PresetCfg

from .camera_cfg import (
    BaseTiledCameraCfg,
    DuoCameraObservationsCfg,
    SingleCameraObservationsCfg,
    WristTiledCameraCfg,
)
from .kuka_allegro_env_cfg import (
    KukaAllegroLiftEnvCfg,
    KukaAllegroReorientEnvCfg,
    KukaAllegroSceneCfg,
)

_SCENE_KWARGS = {"num_envs": 4096, "env_spacing": 3, "replicate_physics": True}


@dataclass
class SingleCameraSceneCfg(KukaAllegroSceneCfg):
    """KukaAllegro scene with a single base-mounted camera."""

    base_camera: CameraCfg = config_field(BaseTiledCameraCfg())


@dataclass
class DuoCameraSceneCfg(KukaAllegroSceneCfg):
    """KukaAllegro scene with base-mounted and wrist-mounted cameras."""

    base_camera: CameraCfg = config_field(BaseTiledCameraCfg())
    wrist_camera: CameraCfg = config_field(WristTiledCameraCfg())


def _camera_env(base_cls, scene_cls, obs_cls):
    """Build a camera env config by swapping a camera scene and image observations onto a state env."""
    return base_cls(scene=scene_cls(**_SCENE_KWARGS), observations=obs_cls())


@dataclass
class KukaAllegroReorientCameraEnvCfg(PresetCfg):
    single_camera: Any = config_field(
        _camera_env(KukaAllegroReorientEnvCfg, SingleCameraSceneCfg, SingleCameraObservationsCfg)
    )
    duo_camera: Any = config_field(_camera_env(KukaAllegroReorientEnvCfg, DuoCameraSceneCfg, DuoCameraObservationsCfg))
    default: Any = config_field(single_camera)


@dataclass
class KukaAllegroLiftCameraEnvCfg(PresetCfg):
    single_camera: Any = config_field(
        _camera_env(KukaAllegroLiftEnvCfg, SingleCameraSceneCfg, SingleCameraObservationsCfg)
    )
    duo_camera: Any = config_field(_camera_env(KukaAllegroLiftEnvCfg, DuoCameraSceneCfg, DuoCameraObservationsCfg))
    default: Any = config_field(single_camera)
