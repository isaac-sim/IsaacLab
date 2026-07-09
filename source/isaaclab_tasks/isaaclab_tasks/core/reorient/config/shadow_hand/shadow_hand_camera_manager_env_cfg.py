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
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_camera_env_cfg import (
    ShadowHandCameraEnvCfg,
    ShadowHandTiledCameraCfg,
    validate_shadow_hand_camera_settings,
)
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_manager_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    EventCfg,
    FullStateWithoutActionCfg,
    RewardsCfg,
    ShadowHandManagerEnvCfg,
    TerminationsCfg,
    _ShadowHandManagerSceneCfg,
)
from isaaclab_tasks.core.reorient.reorient_task_constants import CAMERA_GOAL_MARKER_POSITION
from isaaclab_tasks.utils import PresetCfg

_DIRECT_CAMERA_CFG = ShadowHandCameraEnvCfg()
_FINGERTIP_BODY_NAMES = _DIRECT_CAMERA_CFG.fingertip_body_names


@configclass
class _ShadowHandCameraManagerSceneCfg(_ShadowHandManagerSceneCfg):
    """State Manager scene augmented with camera and fingertip-wrench sensors."""

    ground = None
    tiled_camera: ShadowHandTiledCameraCfg = ShadowHandTiledCameraCfg()
    joint_wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class ShadowHandCameraManagerSceneCfg(PresetCfg):
    """Backend-specific camera scene alternatives for training and benchmarking."""

    physx = _ShadowHandCameraManagerSceneCfg(
        num_envs=1225, env_spacing=2.0, replicate_physics=True, clone_in_fabric=True
    )
    newton_mjwarp = _ShadowHandCameraManagerSceneCfg(
        num_envs=1225, env_spacing=2.0, replicate_physics=True, clone_in_fabric=False
    )
    ovphysx = _ShadowHandCameraManagerSceneCfg(
        num_envs=1225, env_spacing=2.0, replicate_physics=True, clone_in_fabric=True
    )
    default = physx


@configclass
class ShadowHandCameraManagerPlaySceneCfg(PresetCfg):
    """Reduced backend-specific camera scenes for checkpoint playback."""

    physx = _ShadowHandCameraManagerSceneCfg(num_envs=64, env_spacing=2.0, replicate_physics=True, clone_in_fabric=True)
    newton_mjwarp = _ShadowHandCameraManagerSceneCfg(
        num_envs=64, env_spacing=2.0, replicate_physics=True, clone_in_fabric=False
    )
    ovphysx = _ShadowHandCameraManagerSceneCfg(
        num_envs=64, env_spacing=2.0, replicate_physics=True, clone_in_fabric=True
    )
    default = physx


@configclass
class CameraPolicyCfg(FullStateWithoutActionCfg):
    """Direct-compatible 191-dimensional camera actor observation."""

    last_action = ObsTerm(func=mdp.reorient_last_action, params={"action_name": "joint_pos"})
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
class CameraCriticCfg(FullStateWithoutActionCfg):
    """Direct-compatible 214-dimensional asymmetric camera critic state."""

    fingertip_wrench = ObsTerm(
        func=mdp.fingertip_wrench,
        scale=_DIRECT_CAMERA_CFG.force_torque_obs_scale,
        params={"sensor_cfg": SceneEntityCfg("joint_wrench", body_names=_FINGERTIP_BODY_NAMES, preserve_order=False)},
    )
    last_action = ObsTerm(func=mdp.reorient_last_action, params={"action_name": "joint_pos"})
    camera_features = ObsTerm(func=mdp.shadow_hand_camera_cached_features)


@configclass
class CameraObservationsCfg:
    """Camera actor and asymmetric critic observation groups."""

    policy: CameraPolicyCfg = CameraPolicyCfg()
    critic: CameraCriticCfg = CameraCriticCfg()


@configclass
class ShadowHandCameraManagerEnvCfg(ShadowHandManagerEnvCfg):
    """Manager-based camera task with exact Direct dynamics and observations."""

    scene: ShadowHandCameraManagerSceneCfg = ShadowHandCameraManagerSceneCfg()
    observations: CameraObservationsCfg = CameraObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    feature_extractor: FeatureExtractorCfg = FeatureExtractorCfg()

    def __post_init__(self):
        super().__post_init__()
        # camera tasks display the goal inside the tiled camera's frustum
        self.commands.object_pose.fixed_marker_pos = CAMERA_GOAL_MARKER_POSITION
        self.observations.policy.camera_features.params["feature_extractor_cfg"] = self.feature_extractor

    def validate_config(self):
        """Check every unresolved scene alternative or the selected camera pipeline."""
        if isinstance(self.scene, PresetCfg):
            scenes = (self.scene.physx, self.scene.newton_mjwarp, self.scene.ovphysx)
        else:
            scenes = (self.scene,)
        for scene in scenes:
            validate_shadow_hand_camera_settings(scene.tiled_camera, self.feature_extractor)


@configclass
class ShadowHandCameraManagerPlayEnvCfg(ShadowHandCameraManagerEnvCfg):
    """Manager camera task configured for checkpoint playback."""

    scene: ShadowHandCameraManagerPlaySceneCfg = ShadowHandCameraManagerPlaySceneCfg()
    feature_extractor: FeatureExtractorCfg = FeatureExtractorCfg(train=False, load_checkpoint=True)
