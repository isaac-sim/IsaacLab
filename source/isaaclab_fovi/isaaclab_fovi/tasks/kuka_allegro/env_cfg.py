# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kuka-Allegro lift and reorientation with a foveated base camera.

The stock single-camera tasks with the base camera foveated and two gaze actions. The teacher looks at the object:
the goal is a commanded pose, given to the policy as state and not drawn in the image, so there is nothing else to
look at.
"""

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.core.lift.config.kuka_allegro.camera_cfg import BaseTiledCameraCfg, SingleCameraObservationsCfg
from isaaclab_tasks.core.lift.config.kuka_allegro.kuka_allegro_camera_env_cfg import SingleCameraSceneCfg
from isaaclab_tasks.core.lift.config.kuka_allegro.kuka_allegro_env_cfg import (
    KukaAllegroLiftEnvCfg,
    KukaAllegroRelJointPosActionCfg,
    KukaAllegroReorientEnvCfg,
)

from ... import mdp
from ..active_gaze import GazeObservationsCfg, foveated_presets, oracle_observations, select_presets

BASE_CAMERAS = foveated_presets(BaseTiledCameraCfg())
"""The base camera presets that are cameras (not ray casters), foveated."""


@configclass
class SceneCfg(SingleCameraSceneCfg):
    base_camera = BASE_CAMERAS


@configclass
class ActionsCfg(KukaAllegroRelJointPosActionCfg):
    # The gaze must be the last action term.
    gaze = mdp.GazeActionCfg(asset_name="base_camera")


@configclass
class ObservationsCfg(SingleCameraObservationsCfg):
    @configclass
    class BaseImageObsCfg(SingleCameraObservationsCfg.BaseImageObsCfg):
        object_observation_b = select_presets(
            SingleCameraObservationsCfg.BaseImageObsCfg().object_observation_b, BASE_CAMERAS.__dataclass_fields__
        )

    base_image = BaseImageObsCfg()
    gaze = GazeObservationsCfg()
    oracle = oracle_observations(SceneEntityCfg("object"))


@configclass
class KukaAllegroLiftFoviEnvCfg(KukaAllegroLiftEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=3)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Moving the gaze is free; the action penalty applies to the hand and arm only.
        self.rewards.action_l2.func = mdp.motor_action_l2


@configclass
class KukaAllegroReorientFoviEnvCfg(KukaAllegroReorientEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=3)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Moving the gaze is free; the action penalty applies to the hand and arm only.
        self.rewards.action_l2.func = mdp.motor_action_l2
