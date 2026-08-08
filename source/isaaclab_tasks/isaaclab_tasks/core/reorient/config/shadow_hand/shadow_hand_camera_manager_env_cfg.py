# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based counterpart of the Shadow Hand camera reorientation task."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import JointWrenchSensorCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.reorient.mdp as mdp
from isaaclab_tasks.core.reorient.config.shadow_hand.feature_extractor import FeatureExtractorCfg
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_direct_camera_env_cfg import (
    ShadowHandTiledCameraCfg,
    validate_shadow_hand_camera_settings,
)
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_manager_env_cfg import (
    FullStateObsCfg,
    ShadowHandManagerEnvCfg,
    ShadowHandManagerSceneCfg,
)

from isaaclab_assets.robots.shadow_hand import SHADOW_FINGERTIP_BODY_NAMES


@configclass
class ShadowHandCameraManagerSceneCfg(ShadowHandManagerSceneCfg):
    """State Manager scene augmented with camera and fingertip-wrench sensors."""

    num_envs = 1225
    env_spacing = 2.0

    ground = None
    tiled_camera: ShadowHandTiledCameraCfg = ShadowHandTiledCameraCfg()
    joint_wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class CameraPolicyCfg(FullStateObsCfg):
    """Direct-compatible 191-dimensional camera actor observation."""

    last_action = ObsTerm(func=mdp.last_action, params={"action_name": "joint_pos"})
    camera_features = ObsTerm(
        func=mdp.ShadowHandCameraFeatures,
        params={
            "feature_extractor_cfg": FeatureExtractorCfg(),
            "sensor_cfg": SceneEntityCfg("tiled_camera"),
            "object_cfg": SceneEntityCfg("object"),
        },
    )
    goal_keypoints = ObsTerm(func=mdp.shadow_hand_goal_keypoints, params={"command_name": "object_pose"})

    def __post_init__(self):
        super().__post_init__()
        # Camera actor observations infer object state from pixels. These five
        # privileged state terms are present only in the critic.
        self.object_pos = None
        self.object_quat = None
        self.object_lin_vel = None
        self.object_ang_vel = None
        self.goal_quat_diff = None


@configclass
class CameraCriticCfg(FullStateObsCfg):
    """Direct-compatible 214-dimensional asymmetric camera critic state."""

    fingertip_wrench = ObsTerm(
        func=mdp.body_incoming_wrench,
        scale=10.0,
        params={
            "sensor_cfg": SceneEntityCfg("joint_wrench", body_names=SHADOW_FINGERTIP_BODY_NAMES, preserve_order=False)
        },
    )
    last_action = ObsTerm(func=mdp.last_action, params={"action_name": "joint_pos"})
    camera_features = ObsTerm(func=mdp.shadow_hand_camera_cached_features)


@configclass
class CameraObservationsCfg:
    """Camera actor and asymmetric critic observation groups."""

    policy: CameraPolicyCfg = CameraPolicyCfg()
    critic: CameraCriticCfg = CameraCriticCfg()


@configclass
class ShadowHandCameraManagerEnvCfg(ShadowHandManagerEnvCfg):
    """Manager-based camera task with exact Direct dynamics and observations."""

    # only the fields that differ from ShadowHandManagerEnvCfg are overridden
    scene: ShadowHandCameraManagerSceneCfg = ShadowHandCameraManagerSceneCfg()
    observations: CameraObservationsCfg = CameraObservationsCfg()
    feature_extractor: FeatureExtractorCfg = FeatureExtractorCfg()

    def __post_init__(self):
        super().__post_init__()
        # camera tasks display the goal inside the tiled camera's frustum
        # goal cube must sit inside the tiled camera's frustum
        self.commands.object_pose.fixed_marker_pos = (-0.2, 0.1, 0.6)
        self.observations.policy.camera_features.params["feature_extractor_cfg"] = self.feature_extractor
        # Only the Isaac RTX tiled camera renders the default modalities, so the camera
        # task defaults to it. Newton stays selectable via ``physics=newton_mjwarp``.
        for backend_cfg in (self.sim.physics, self.scene.robot):
            backend_cfg.default = backend_cfg.isaacsim_physx

    def validate_config(self):
        """Check the camera pipeline against the feature extractor it feeds."""
        validate_shadow_hand_camera_settings(self.scene.tiled_camera, self.feature_extractor)

    def play_mode(self):
        super().play_mode()
        # the tiled camera needs more environments than the shared play default
        self.scene.num_envs = 64
        # mutate rather than replace: subclasses may have disabled the CNN
        self.feature_extractor.train = False
        self.feature_extractor.load_checkpoint = True
        self.observations.policy.camera_features.params["feature_extractor_cfg"] = self.feature_extractor
