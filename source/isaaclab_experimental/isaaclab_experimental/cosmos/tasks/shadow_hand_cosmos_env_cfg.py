# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shadow Hand camera reorientation tasks with depth-guided Cosmos RGB observations."""

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab_tasks.core.reorient.config.shadow_hand.feature_extractor import FeatureExtractorCfg
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_camera_manager_env_cfg import (
    ShadowHandCameraManagerEnvCfg,
    ShadowHandCameraManagerSceneCfg,
)
from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_direct_camera_env_cfg import (
    ShadowHandCameraEnvCfg,
    ShadowHandCameraSceneCfg,
    _ShadowHandBaseTiledCameraCfg,
)

from .. import CosmosModelCfg, CosmosTransferProcessorCfg, cosmos_camera, episode_frame_budget

_COSMOS_MODEL_CFG = CosmosModelCfg(
    prompt=(
        "A close-up overhead view of a robotic Shadow Hand reorienting a small colored cube, "
        "with realistic metallic fingers, natural lighting, and a dark background."
    ),
)
# The Cosmos tasks feed the 640 x 640 generated image to the CNN, so the camera already uses a Cosmos canvas.
_COSMOS_BASE_CAMERA_CFG = _ShadowHandBaseTiledCameraCfg(data_types=["rgb"], height=640, width=640)

SHADOW_HAND_COSMOS_CAMERA_CFG = cosmos_camera(_COSMOS_BASE_CAMERA_CFG, _COSMOS_MODEL_CFG, near=0.1, far=1.5)
"""Camera rendering depth that Cosmos turns into the published ``rgb``; captures every environment step."""


def shadow_hand_cosmos_settings(env_cfg) -> CosmosModelCfg:
    """Return the Cosmos settings of the camera, the object its Cosmos step uses (not a copy).

    Shared by the Direct and Manager-based tasks.

    Args:
        env_cfg: Environment configuration with ``scene.tiled_camera``.
    """
    return _cosmos_transfer(env_cfg).backend


def set_shadow_hand_cosmos_settings(env_cfg, settings: CosmosModelCfg) -> None:
    """Make the camera's Cosmos step use ``settings``."""
    _cosmos_transfer(env_cfg).backend = settings


def _cosmos_transfer(env_cfg) -> CosmosTransferProcessorCfg:
    """Return the camera's only Cosmos step."""
    transfers = [
        cfg for cfg in env_cfg.scene.tiled_camera.tensor_processors if isinstance(cfg, CosmosTransferProcessorCfg)
    ]
    if len(transfers) != 1:
        raise ValueError("The Shadow Hand Cosmos task requires one Cosmos step among the camera's tensor processors.")
    return transfers[0]


def validate_shadow_hand_cosmos_task(env_cfg) -> None:
    """Check a Shadow Hand Cosmos task against the Cosmos service's stream limits.

    Shared by the Direct and Manager-based tasks.

    Args:
        env_cfg: Environment configuration with ``scene.tiled_camera`` and ``feature_extractor``.

    Raises:
        ValueError: If the scene, rendering, supervision cadence, or episode length breaks a Cosmos limit.
    """
    if not env_cfg.scene.lazy_sensor_update:
        raise ValueError(
            "The Shadow Hand Cosmos task requires scene.lazy_sensor_update=True so generated "
            "images and feature-extractor supervision stay aligned with camera captures."
        )
    camera = env_cfg.scene.tiled_camera
    # Only the OVRTX renderer configuration has this setting.
    if getattr(camera.renderer_cfg, "async_rendering", False):
        raise ValueError(
            "The Shadow Hand Cosmos task requires synchronous OVRTX rendering; set "
            "scene.tiled_camera.renderer_cfg.async_rendering=False "
            "(env.scene.tiled_camera.renderer_cfg.async_rendering=False on the command line)."
        )
    if env_cfg.feature_extractor.train and env_cfg.scene.num_envs > 1 and camera.update_period > 0:
        # Environments that reset at different times then capture on different steps, so a generated image can come
        # from an earlier capture than the step it appears on, while training pairs it with the current pose.
        raise ValueError(
            "Training the Shadow Hand Cosmos task with several environments requires a capture every step; set "
            "scene.tiled_camera.update_period=0 (env.scene.tiled_camera.update_period=0 on the command line)."
        )
    transfer = _cosmos_transfer(env_cfg)
    if env_cfg.feature_extractor.image_update_frames != transfer.update_frames:
        raise ValueError(
            "The Shadow Hand Cosmos task requires feature_extractor.image_update_frames "
            f"to match the Cosmos update_frames ({transfer.update_frames})."
        )
    _, episode_frames, budget = episode_frame_budget(env_cfg, camera.update_period)
    if transfer.backend.max_episode_frames is None:
        transfer.backend.max_episode_frames = budget
    frame_budget = transfer.backend.max_episode_frames
    if episode_frames > frame_budget:
        raise ValueError(
            f"The Shadow Hand Cosmos task needs up to {episode_frames} camera captures per episode, "
            f"but the Cosmos frame budget is {frame_budget}. Shorten episode_length_s or increase "
            "scene.tiled_camera.update_period (env.scene.tiled_camera.update_period on the command line)."
        )


@configclass
class ShadowHandCameraCosmosEnvCfg(ShadowHandCameraEnvCfg):
    """Direct Shadow Hand task with depth-guided Cosmos RGB observations.

    Registered as ``Isaac-Reorient-Cube-Shadow-Camera-Cosmos-Direct``. Start the Cosmos service separately. The
    task uses one environment unless ``--num_envs`` sets more. The camera captures every environment step, and
    Cosmos publishes an initial frame followed by four-frame chunks. The feature extractor trains on the generated
    RGB; playback requires its checkpoint from a Cosmos run.
    """

    scene: ShadowHandCameraSceneCfg = ShadowHandCameraSceneCfg(num_envs=1, tiled_camera=SHADOW_HAND_COSMOS_CAMERA_CFG)
    feature_extractor: FeatureExtractorCfg = FeatureExtractorCfg(pretrained_checkpoint=None, image_update_frames=4)

    @property
    def cosmos(self) -> CosmosModelCfg:
        """The camera's Cosmos settings; overrides such as ``env.cosmos.prompt=...`` set them on the camera."""
        return shadow_hand_cosmos_settings(self)

    @cosmos.setter
    def cosmos(self, settings: CosmosModelCfg):
        # Also lets the configclass copier, which copies inherited properties like fields, set it in subclasses.
        set_shadow_hand_cosmos_settings(self, settings)

    def validate_config(self):
        """Check the scene and episode fit the Cosmos service's stream limits."""
        super().validate_config()
        validate_shadow_hand_cosmos_task(self)
        if self.max_consecutive_success != 0:
            raise ValueError(
                "The Shadow Hand Cosmos task requires max_consecutive_success=0 so goal successes "
                "do not extend the camera episode beyond its frame budget."
            )

    def play_mode(self):
        """Load this run's feature extractor and retain the single-camera scene."""
        super().play_mode()
        self.scene.num_envs = 1


@configclass
class ShadowHandCameraManagerCosmosEnvCfg(ShadowHandCameraManagerEnvCfg):
    """Manager-based Shadow Hand task with depth-guided Cosmos RGB observations.

    Registered as ``Isaac-Reorient-Cube-Shadow-Camera-Cosmos``. The camera runs the same Cosmos processors as the
    Direct task, and the policy's camera observation term reads the generated ``rgb``. Start the Cosmos service
    first.
    """

    scene: ShadowHandCameraManagerSceneCfg = ShadowHandCameraManagerSceneCfg(
        num_envs=1, tiled_camera=SHADOW_HAND_COSMOS_CAMERA_CFG
    )
    feature_extractor: FeatureExtractorCfg = FeatureExtractorCfg(pretrained_checkpoint=None, image_update_frames=4)

    @property
    def cosmos(self) -> CosmosModelCfg:
        """The camera's Cosmos settings; overrides such as ``env.cosmos.prompt=...`` set them on the camera."""
        return shadow_hand_cosmos_settings(self)

    @cosmos.setter
    def cosmos(self, settings: CosmosModelCfg):
        # Also lets the configclass copier, which copies inherited properties like fields, set it in subclasses.
        set_shadow_hand_cosmos_settings(self, settings)

    def validate_config(self):
        """Check the scene and episode fit the Cosmos service's stream limits."""
        super().validate_config()
        validate_shadow_hand_cosmos_task(self)

    def play_mode(self):
        """Load this run's feature extractor and retain the single-camera scene."""
        super().play_mode()
        self.scene.num_envs = 1
