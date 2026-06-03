# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

import isaaclab_tasks.core.cartpole.mdp as mdp
from isaaclab_tasks.core.cartpole.cartpole_manager_env_cfg import CartpoleEnvCfg, CartpoleSceneCfg
from isaaclab_tasks.utils import PresetCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

##
# Camera presets
##


@configclass
class CartpoleTiledCameraCfg(PresetCfg):
    """Tiled-camera presets, one per rendered data type.

    Each variant selects its rendering backend (RTX, OmniverseRTX, Newton + Warp) through the
    nested :attr:`~BaseCartpoleTiledCameraCfg.renderer_cfg` preset, so a single ``presets=`` selector
    can pick both the data type and the backend.
    """

    @configclass
    class BaseCartpoleTiledCameraCfg(CameraCfg):
        prim_path: str = "/World/envs/env_.*/Camera"
        offset: CameraCfg.OffsetCfg = CameraCfg.OffsetCfg(
            pos=(-5.0, 0.0, 2.0), rot=(0.0, 0.0, 0.0, 1.0), convention="world"
        )
        data_types: list[str] = []
        spawn: sim_utils.PinholeCameraCfg = sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        )
        width: int = 100
        height: int = 100
        renderer_cfg: MultiBackendRendererCfg = MultiBackendRendererCfg()

    default = BaseCartpoleTiledCameraCfg(data_types=["rgb"])
    depth = BaseCartpoleTiledCameraCfg(data_types=["depth"])
    albedo = BaseCartpoleTiledCameraCfg(data_types=["albedo"])
    semantic_segmentation = BaseCartpoleTiledCameraCfg(data_types=["semantic_segmentation"])
    simple_shading_constant_diffuse = BaseCartpoleTiledCameraCfg(data_types=["simple_shading_constant_diffuse"])
    simple_shading_diffuse_mdl = BaseCartpoleTiledCameraCfg(data_types=["simple_shading_diffuse_mdl"])
    simple_shading_full_mdl = BaseCartpoleTiledCameraCfg(data_types=["simple_shading_full_mdl"])
    rgb = default


##
# Scene definition
##


@configclass
class CartpoleCameraSceneCfg(CartpoleSceneCfg):
    """Cartpole scene with a selectable tiled camera."""

    tiled_camera: CartpoleTiledCameraCfg = CartpoleTiledCameraCfg()


##
# MDP settings
##


def image_observations_cfg(data_type: str):
    """Build a single-camera-image policy observation group.

    Args:
        data_type: Camera data type to read from the tiled camera (e.g. ``"rgb"``, ``"depth"``).

    Returns:
        An observations config whose ``policy`` group emits the permuted camera image.
    """

    @configclass
    class ImageObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            image = ObsTerm(
                func=mdp.image,
                params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": data_type, "permute": True},
            )

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        policy: ObsGroup = PolicyCfg()

    return ImageObservationsCfg()


@configclass
class ResNet18ObservationCfg:
    """Observation specifications for the MDP."""

    @configclass
    class ResNet18FeaturesCameraPolicyCfg(ObsGroup):
        """Observations for policy group with features extracted from RGB images with a frozen ResNet18."""

        image = ObsTerm(
            func=mdp.image_features,
            params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb", "model_name": "resnet18"},
        )

    policy: ObsGroup = ResNet18FeaturesCameraPolicyCfg()


@configclass
class TheiaTinyObservationCfg:
    """Observation specifications for the MDP."""

    @configclass
    class TheiaTinyFeaturesCameraPolicyCfg(ObsGroup):
        """Observations for policy group with features extracted from RGB images with a frozen Theia-Tiny Transformer"""

        image = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("tiled_camera"),
                "data_type": "rgb",
                "model_name": "theia-tiny-patch16-224-cddsv",
                "model_device": "cuda:0",
            },
        )

    policy: ObsGroup = TheiaTinyFeaturesCameraPolicyCfg()


##
# Environment configuration
##


@configclass
class CartpoleCameraEnvCfg(PresetCfg):
    """Cartpole environment with a selectable camera observation pipeline.

    A single ``presets=`` selector cascades through this preset: it picks the observation pipeline
    here and, via :attr:`CartpoleCameraSceneCfg.tiled_camera`, the matching camera data type and
    rendering backend. The feature-extractor variants (``resnet18``, ``theia_tiny``) operate on RGB
    images, so they fall back to the default camera.
    """

    @configclass
    class BaseCartpoleCameraEnvCfg(CartpoleEnvCfg):
        """Camera variant of :class:`CartpoleEnvCfg` -- only the fields that differ are overridden."""

        # scene: fewer, more-spaced envs so each camera renders cleanly
        scene: CartpoleCameraSceneCfg = CartpoleCameraSceneCfg(num_envs=512, env_spacing=20.0)

        def __post_init__(self):
            super().__post_init__()
            # remove ground as it obstructs the camera
            self.scene.ground = None
            # viewer settings
            self.viewer.eye = (7.0, 0.0, 2.5)
            self.viewer.lookat = (0.0, 0.0, 2.5)

    rgb = BaseCartpoleCameraEnvCfg(observations=image_observations_cfg("rgb"))
    depth = BaseCartpoleCameraEnvCfg(observations=image_observations_cfg("depth"))
    albedo = BaseCartpoleCameraEnvCfg(observations=image_observations_cfg("albedo"))
    semantic_segmentation = BaseCartpoleCameraEnvCfg(observations=image_observations_cfg("semantic_segmentation"))
    simple_shading_constant_diffuse = BaseCartpoleCameraEnvCfg(
        observations=image_observations_cfg("simple_shading_constant_diffuse")
    )
    simple_shading_diffuse_mdl = BaseCartpoleCameraEnvCfg(
        observations=image_observations_cfg("simple_shading_diffuse_mdl")
    )
    simple_shading_full_mdl = BaseCartpoleCameraEnvCfg(observations=image_observations_cfg("simple_shading_full_mdl"))
    resnet18 = BaseCartpoleCameraEnvCfg(observations=ResNet18ObservationCfg())
    theia_tiny = BaseCartpoleCameraEnvCfg(observations=TheiaTinyObservationCfg())
    default = rgb
