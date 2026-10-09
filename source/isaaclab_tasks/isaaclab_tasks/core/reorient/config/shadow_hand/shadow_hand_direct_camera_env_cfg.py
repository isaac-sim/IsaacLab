# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the direct-workflow Shadow Hand camera reorientation environment."""

from __future__ import annotations

import math

from isaaclab_experimental.cosmos import CosmosModelCfg, CosmosTransferModifierCfg, cosmos_camera
from isaaclab_ov.renderers import OVRTXRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.renderers import RendererCfg
from isaaclab.sensors import CameraCfg, JointWrenchSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_tasks.utils import PresetCfg, preset
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from .feature_extractor import FeatureExtractorCfg
from .shadow_hand_direct_env_cfg import (
    ShadowHandEnvCfg,
    ShadowHandSceneCfg,
)

_PRETRAINED_CHECKPOINT_DIR = f"{ISAACLAB_NUCLEUS_DIR}/PretrainedCheckpoints/rsl_rl"
_DIRECT_NEWTON_FEATURE_EXTRACTOR_CHECKPOINT = (
    f"{_PRETRAINED_CHECKPOINT_DIR}/"
    "Isaac-Reorient-Cube-Shadow-Camera-Direct_newtonmjwarp_newton_rsl_rl_feature_extractor.pth"
)
_DIRECT_PHYSX_FEATURE_EXTRACTOR_CHECKPOINT = (
    f"{_PRETRAINED_CHECKPOINT_DIR}/Isaac-Reorient-Cube-Shadow-Camera-Direct_physx_rtx_rsl_rl_feature_extractor.pth"
)


def validate_shadow_hand_camera_settings(
    tiled_camera: CameraCfg,
    feature_extractor: FeatureExtractorCfg,
) -> None:
    """Validate one concrete Shadow Hand camera pipeline."""
    if not isinstance(tiled_camera, CameraCfg):
        raise TypeError(
            f"Shadow Hand camera validation requires a concrete CameraCfg, got {type(tiled_camera).__name__}."
        )
    renderer_cfg = tiled_camera.renderer_cfg
    if renderer_cfg is not None and not isinstance(renderer_cfg, RendererCfg):
        raise TypeError(
            f"Shadow Hand camera validation requires a concrete RendererCfg or None, got {type(renderer_cfg).__name__}."
        )

    non_depth_data_types = set(tiled_camera.data_types).difference(
        {"depth", "distance_to_image_plane", "distance_to_camera"}
    )
    if tiled_camera.data_types and not non_depth_data_types and feature_extractor.enabled:
        raise ValueError(
            "Depth-only camera data type is intended for benchmarking only. "
            "The keypoint-regression CNN cannot be meaningfully trained from depth alone. "
            "Disable the feature extractor with 'feature_extractor.enabled=False' "
            "(e.g. use IsaacContrib-Reorient-Cube-Shadow-Camera-Benchmark-Direct), "
            "or choose a data type that includes colour, e.g. presets=rgb."
        )


@configclass
class _ShadowHandBaseTiledCameraCfg(CameraCfg):
    """Base camera configuration for the shadow hand vision environment.

    This is an internal config used by :class:`ShadowHandTiledCameraCfg` presets and
    by derived env configs that hard-code a specific data type. It embeds
    :class:`~isaaclab_tasks.utils.MultiBackendRendererCfg` so the renderer backend can
    still be selected via the ``presets`` CLI argument.
    """

    prim_path: str = "{ENV_REGEX_NS}/Camera"
    offset: CameraCfg.OffsetCfg = CameraCfg.OffsetCfg(
        pos=(0, -0.35, 1.0), rot=(0.0, 0.7071, 0.0, 0.7071), convention="world"
    )
    data_types: list[str] = []
    spawn: sim_utils.PinholeCameraCfg = sim_utils.PinholeCameraCfg(
        spawn_path="/World/envs/env_0/Camera",
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 20.0),
    )
    width: int = 120
    height: int = 120
    renderer_cfg: MultiBackendRendererCfg = MultiBackendRendererCfg()


@configclass
class ShadowHandTiledCameraCfg(PresetCfg):
    """Camera data-type presets for the shadow hand vision environment.

    Each preset selects which image modalities are captured. The selected data types must
    match :attr:`FeatureExtractorCfg.data_types` so the CNN receives the expected channels.

    Select a data-type preset via the ``presets`` CLI argument, e.g.::

        presets = rgb  # RGB only (3 channels)
        presets = albedo  # albedo (3 channels)
        presets = simple_shading_constant_diffuse  # simple shading, constant diffuse (3 channels)

    Renderer and data-type presets can be combined::

        presets = newton_renderer, rgb
    """

    default: _ShadowHandBaseTiledCameraCfg = _ShadowHandBaseTiledCameraCfg(
        data_types=["rgb", "depth", "semantic_segmentation"]
    )
    """Default: RGB + depth + semantic segmentation (7 CNN input channels)."""

    full: _ShadowHandBaseTiledCameraCfg = _ShadowHandBaseTiledCameraCfg(
        data_types=["rgb", "depth", "semantic_segmentation"]
    )
    """Full modalities: RGB + depth + semantic segmentation (7 channels). Alias for default."""

    rgb: _ShadowHandBaseTiledCameraCfg = _ShadowHandBaseTiledCameraCfg(data_types=["rgb"])
    """RGB only (3 CNN input channels)."""

    albedo: _ShadowHandBaseTiledCameraCfg = _ShadowHandBaseTiledCameraCfg(data_types=["albedo"])
    """Albedo (3 CNN input channels)."""

    simple_shading_constant_diffuse: _ShadowHandBaseTiledCameraCfg = _ShadowHandBaseTiledCameraCfg(
        data_types=["simple_shading_constant_diffuse"]
    )
    """Simple shading with constant diffuse (3 CNN input channels)."""

    simple_shading_diffuse_mdl: _ShadowHandBaseTiledCameraCfg = _ShadowHandBaseTiledCameraCfg(
        data_types=["simple_shading_diffuse_mdl"]
    )
    """Simple shading with diffuse MDL (3 CNN input channels)."""

    simple_shading_full_mdl: _ShadowHandBaseTiledCameraCfg = _ShadowHandBaseTiledCameraCfg(
        data_types=["simple_shading_full_mdl"]
    )
    """Simple shading with full MDL (3 CNN input channels)."""

    depth: _ShadowHandBaseTiledCameraCfg = _ShadowHandBaseTiledCameraCfg(data_types=["depth"])
    """Depth only (1 channel).

    .. warning::
        This preset is intended for **benchmarking only**. The keypoint-regression CNN
        cannot be meaningfully trained from depth alone. Use it with the contributed
        benchmark task, which disables the feature extractor, to measure pure
        depth-rendering throughput, e.g.::

            presets=depth          # depth rendering with the default Newton renderer
            presets=depth,ovrtx    # depth rendering with OVRTX renderer
    """

    semantic_segmentation: _ShadowHandBaseTiledCameraCfg = _ShadowHandBaseTiledCameraCfg(
        data_types=["semantic_segmentation"]
    )
    """Semantic segmentation (3 CNN input channels)."""


@configclass
class ShadowHandCameraSceneCfg(ShadowHandSceneCfg):
    """Shadow Hand scene with camera and fingertip-wrench sensors."""

    num_envs = 1225
    env_spacing = 2.0
    ground = None
    joint_wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")
    tiled_camera: ShadowHandTiledCameraCfg = ShadowHandTiledCameraCfg()


@configclass
class ShadowHandCameraEnvCfg(ShadowHandEnvCfg):
    """Configuration for the direct-workflow Shadow Hand camera reorientation environment."""

    # scene
    scene: ShadowHandCameraSceneCfg = ShadowHandCameraSceneCfg()

    feature_extractor: FeatureExtractorCfg = FeatureExtractorCfg(
        pretrained_checkpoint=preset(  # type: ignore[arg-type]
            default=_DIRECT_NEWTON_FEATURE_EXTRACTOR_CHECKPOINT,
            newton_mjwarp=_DIRECT_NEWTON_FEATURE_EXTRACTOR_CHECKPOINT,
            isaacsim_physx=_DIRECT_PHYSX_FEATURE_EXTRACTOR_CHECKPOINT,
            ovphysx=_DIRECT_PHYSX_FEATURE_EXTRACTOR_CHECKPOINT,
            physx=_DIRECT_PHYSX_FEATURE_EXTRACTOR_CHECKPOINT,
        )
    )

    # env
    observation_space = 164 + 27  # state observation + vision CNN embedding
    state_space = 187 + 27  # asymmetric states + vision CNN embedding

    def validate_config(self):
        """Check renderer/data-type and feature-extractor compatibility."""
        validate_shadow_hand_camera_settings(self.scene.tiled_camera, self.feature_extractor)

    def play_mode(self):
        # play-mode overrides of parent
        super().play_mode()

        # scene
        self.scene.num_envs = 64
        # inference for CNN
        self.feature_extractor.train = False
        self.feature_extractor.load_checkpoint = True


_COSMOS_DEPTH_INPUT = "distance_to_image_plane"
_COSMOS_MODEL_CFG = CosmosModelCfg(
    modality="depth",
    prompt=(
        "A close-up overhead view of a robotic Shadow Hand reorienting a small colored cube, "
        "with realistic metallic fingers, natural lighting, and a dark background."
    ),
)
# The Cosmos presets feed the 640 x 640 generated image to the CNN, so the camera already uses a Cosmos canvas.
_COSMOS_BASE_CAMERA_CFG = _ShadowHandBaseTiledCameraCfg(data_types=["rgb"], height=640, width=640, update_period=0.1)

SHADOW_HAND_COSMOS_CAMERA_CFG = cosmos_camera(_COSMOS_BASE_CAMERA_CFG, _COSMOS_MODEL_CFG, near=0.1, far=1.5)
"""Camera rendering depth that Cosmos turns into the published ``rgb``; captures every 0.1 s."""


def validate_shadow_hand_cosmos_preset(env_cfg) -> None:
    """Check a Shadow Hand Cosmos preset against the Cosmos service's stream limits.

    Shared by the Direct and Manager-based tasks.

    Args:
        env_cfg: Environment configuration with ``scene.tiled_camera`` and ``feature_extractor``.

    Raises:
        ValueError: If the scene, rendering, supervision cadence, or episode length breaks a Cosmos limit.
    """
    if env_cfg.scene.num_envs != 1:
        raise ValueError("The Shadow Hand Cosmos preset requires one environment; use --num_envs 1.")
    if not env_cfg.scene.lazy_sensor_update:
        raise ValueError(
            "The Shadow Hand Cosmos preset requires scene.lazy_sensor_update=True so generated "
            "images and feature-extractor supervision stay aligned with camera captures."
        )
    camera = env_cfg.scene.tiled_camera
    if isinstance(camera.renderer_cfg, OVRTXRendererCfg) and camera.renderer_cfg.async_rendering:
        raise ValueError(
            "The Shadow Hand Cosmos preset requires synchronous OVRTX rendering; set "
            "scene.tiled_camera.renderer_cfg.async_rendering=False."
        )
    transfers = [
        cfg for cfg in camera.modifiers.get(_COSMOS_DEPTH_INPUT, ()) if isinstance(cfg, CosmosTransferModifierCfg)
    ]
    if len(transfers) != 1:
        raise ValueError("The Shadow Hand Cosmos preset requires one Cosmos step on the camera's depth chain.")
    transfer = transfers[0]
    if env_cfg.feature_extractor.image_update_frames != transfer.update_frames:
        raise ValueError(
            "The Shadow Hand Cosmos preset requires feature_extractor.image_update_frames "
            f"to match the Cosmos update_frames ({transfer.update_frames})."
        )
    capture_period = max(camera.update_period, env_cfg.sim.dt)
    episode_frames = math.ceil(env_cfg.episode_length_s / capture_period) + 1
    frame_budget = transfer.backend.max_episode_frames
    if episode_frames > frame_budget:
        raise ValueError(
            f"The Shadow Hand Cosmos preset needs up to {episode_frames} camera captures per episode, "
            f"but the Cosmos frame budget is {frame_budget}. Shorten episode_length_s or increase "
            "scene.tiled_camera.update_period."
        )


@configclass
class ShadowHandCameraCosmosEnvCfg(ShadowHandCameraEnvCfg):
    """Single-environment Shadow Hand task with depth-guided Cosmos RGB observations.

    Start the Cosmos service separately, then select ``presets=cosmos``. The camera captures at
    10 Hz, and Cosmos publishes an initial frame followed by four-frame chunks. The feature
    extractor trains on the generated RGB; playback requires its checkpoint from a Cosmos run.
    """

    scene: ShadowHandCameraSceneCfg = ShadowHandCameraSceneCfg(num_envs=1, tiled_camera=SHADOW_HAND_COSMOS_CAMERA_CFG)
    feature_extractor: FeatureExtractorCfg = FeatureExtractorCfg(pretrained_checkpoint=None, image_update_frames=4)

    def validate_config(self):
        """Check the scene and episode fit the Cosmos service's stream limits."""
        super().validate_config()
        validate_shadow_hand_cosmos_preset(self)
        if self.max_consecutive_success != 0:
            raise ValueError(
                "The Shadow Hand Cosmos preset requires max_consecutive_success=0 so goal successes "
                "do not extend the camera episode beyond its frame budget."
            )

    def play_mode(self):
        """Load this run's feature extractor and retain the single-camera scene."""
        super().play_mode()
        self.scene.num_envs = 1


@configclass
class ShadowHandCameraEnvPresetCfg(PresetCfg):
    """Select ordinary camera observations or depth-guided Cosmos with ``presets=cosmos``."""

    default: ShadowHandCameraEnvCfg = ShadowHandCameraEnvCfg()
    cosmos: ShadowHandCameraCosmosEnvCfg = ShadowHandCameraCosmosEnvCfg()
