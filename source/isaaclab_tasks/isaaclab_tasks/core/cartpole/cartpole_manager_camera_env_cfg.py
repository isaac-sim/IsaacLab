# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the manager-based cartpole camera environment."""

import math

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.utils import PresetCfg

from . import mdp
from .cartpole_common import CartpoleTiledCameraCfg
from .cartpole_manager_env_cfg import CartpoleEnvCfg, CartpoleSceneCfg, ObservationsCfg

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
        An observations config with camera policy observations and privileged state critic observations.
    """

    @configclass
    class ImageObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            image = ObsTerm(
                func=mdp.CameraImageStack,
                params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": data_type},
            )

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        policy: ObsGroup = PolicyCfg()
        critic: ObsGroup = ObservationsCfg.PolicyCfg()

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
        """Observations for policy group with features extracted from RGB images with a frozen Theia-Tiny model."""

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
        """Camera variant of :class:`CartpoleEnvCfg`; only the fields that differ are overridden."""

        frame_stack: int = 2
        """Number of frames to stack along the channel dimension.

        Values less than two disable stacking.
        """

        # scene: fewer, more-spaced envs so each camera renders cleanly
        scene: CartpoleCameraSceneCfg = CartpoleCameraSceneCfg(num_envs=512, env_spacing=20.0)

        def __post_init__(self):
            super().__post_init__()
            # remove the ground as it obstructs the camera
            self.scene.ground = None
            # reset: smaller initial pole angle than the proprioceptive task
            self.events.reset_pole_position.params["position_range"] = (-0.125 * math.pi, 0.125 * math.pi)
            # visualizer settings
            self.sim.default_visualizer_cfg = VisualizerCfg(eye=(20.0, 20.0, 20.0), lookat=(0.0, 0.0, 0.0))

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
