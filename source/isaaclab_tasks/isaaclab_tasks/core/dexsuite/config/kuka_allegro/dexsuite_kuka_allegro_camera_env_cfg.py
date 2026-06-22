# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera (vision) variants of the KukaAllegro dexsuite reorient/lift tasks.

Each task is exposed as a :class:`~isaaclab_tasks.utils.PresetCfg` whose ``single_camera`` /
``duo_camera`` variants add base (and wrist) cameras plus the matching image observations on top of
the state env config in :mod:`.dexsuite_kuka_allegro_env_cfg`. The camera data type / resolution and
renderer backend remain ``presets=`` selectable through the camera configs.
"""

from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from .camera_cfg import BaseTiledCameraCfg, DuoCameraObservationsCfg, SingleCameraObservationsCfg, WristTiledCameraCfg
from .dexsuite_kuka_allegro_env_cfg import (
    DexsuiteKukaAllegroLiftEnvCfg,
    DexsuiteKukaAllegroReorientEnvCfg,
    KukaAllegroSceneCfg,
)

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


def _camera_env(base_cls, scene_cls, obs_cls):
    """Build a camera env config by swapping a camera scene and image observations onto a state env."""
    return base_cls(scene=scene_cls(**_SCENE_KWARGS), observations=obs_cls())


@configclass
class DexsuiteKukaAllegroReorientCameraEnvCfg(PresetCfg):
    single_camera = _camera_env(DexsuiteKukaAllegroReorientEnvCfg, SingleCameraSceneCfg, SingleCameraObservationsCfg)
    duo_camera = _camera_env(DexsuiteKukaAllegroReorientEnvCfg, DuoCameraSceneCfg, DuoCameraObservationsCfg)
    default = single_camera


@configclass
class DexsuiteKukaAllegroLiftCameraEnvCfg(PresetCfg):
    single_camera = _camera_env(DexsuiteKukaAllegroLiftEnvCfg, SingleCameraSceneCfg, SingleCameraObservationsCfg)
    duo_camera = _camera_env(DexsuiteKukaAllegroLiftEnvCfg, DuoCameraSceneCfg, DuoCameraObservationsCfg)
    default = single_camera
