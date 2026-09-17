# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import dataclass, field
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.visualizers import VisualizerCfg

import isaaclab_tasks.core.cartpole.mdp as mdp
from isaaclab_tasks.core.cartpole.cartpole_manager_env_cfg import CartpoleEnvCfg, CartpoleSceneCfg, ObservationsCfg
from isaaclab_tasks.utils import PresetCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

##
# Camera presets
##


@dataclass
class CartpoleTiledCameraCfg(PresetCfg):
    """Tiled-camera presets, one per rendered data type.

    Each variant selects its rendering backend (RTX, OmniverseRTX, Newton + Warp) through the
    nested :attr:`~BaseCartpoleTiledCameraCfg.renderer_cfg` preset, so a single ``presets=`` selector
    can pick both the data type and the backend.
    """

    @dataclass
    class BaseCartpoleTiledCameraCfg(CameraCfg):
        prim_path: str = "{ENV_REGEX_NS}/Camera"
        offset: CameraCfg.OffsetCfg = field(
            default_factory=lambda: CameraCfg.OffsetCfg(
                pos=(-5.0, 0.0, 2.0), rot=(0.0, 0.0, 0.0, 1.0), convention="world"
            )
        )
        data_types: list[str] = field(default_factory=list)
        spawn: sim_utils.PinholeCameraCfg = field(
            default_factory=lambda: sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
            )
        )
        width: int = 96
        height: int = 96
        renderer_cfg: MultiBackendRendererCfg = field(default_factory=MultiBackendRendererCfg)

    default: Any = field(default_factory=lambda: CartpoleTiledCameraCfg.BaseCartpoleTiledCameraCfg(data_types=["rgb"]))
    depth: Any = field(default_factory=lambda: CartpoleTiledCameraCfg.BaseCartpoleTiledCameraCfg(data_types=["depth"]))
    albedo: Any = field(
        default_factory=lambda: CartpoleTiledCameraCfg.BaseCartpoleTiledCameraCfg(data_types=["albedo"])
    )
    semantic_segmentation: Any = field(
        default_factory=lambda: CartpoleTiledCameraCfg.BaseCartpoleTiledCameraCfg(data_types=["semantic_segmentation"])
    )
    simple_shading_constant_diffuse: Any = field(
        default_factory=lambda: CartpoleTiledCameraCfg.BaseCartpoleTiledCameraCfg(
            data_types=["simple_shading_constant_diffuse"]
        )
    )
    simple_shading_diffuse_mdl: Any = field(
        default_factory=lambda: CartpoleTiledCameraCfg.BaseCartpoleTiledCameraCfg(
            data_types=["simple_shading_diffuse_mdl"]
        )
    )
    simple_shading_full_mdl: Any = field(
        default_factory=lambda: CartpoleTiledCameraCfg.BaseCartpoleTiledCameraCfg(
            data_types=["simple_shading_full_mdl"]
        )
    )
    rgb: Any = field(default_factory=lambda: CartpoleTiledCameraCfg.BaseCartpoleTiledCameraCfg(data_types=["rgb"]))


##
# Scene definition
##


@dataclass
class CartpoleCameraSceneCfg(CartpoleSceneCfg):
    """Cartpole scene with a selectable tiled camera."""

    tiled_camera: CartpoleTiledCameraCfg = field(default_factory=CartpoleTiledCameraCfg)


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

    @dataclass
    class ImageObservationsCfg:
        @dataclass
        class PolicyCfg(ObsGroup):
            image: Any = field(
                default_factory=lambda: ObsTerm(
                    func=mdp.CameraImageStack,
                    params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": data_type},
                )
            )

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        policy: ObsGroup = field(default_factory=PolicyCfg)
        critic: ObsGroup = field(default_factory=ObservationsCfg.PolicyCfg)

    return ImageObservationsCfg()


@dataclass
class ResNet18ObservationCfg:
    """Observation specifications for the MDP."""

    @dataclass
    class ResNet18FeaturesCameraPolicyCfg(ObsGroup):
        """Observations for policy group with features extracted from RGB images with a frozen ResNet18."""

        image: Any = field(
            default_factory=lambda: ObsTerm(
                func=mdp.image_features,
                params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb", "model_name": "resnet18"},
            )
        )

    policy: ObsGroup = field(default_factory=ResNet18FeaturesCameraPolicyCfg)


@dataclass
class TheiaTinyObservationCfg:
    """Observation specifications for the MDP."""

    @dataclass
    class TheiaTinyFeaturesCameraPolicyCfg(ObsGroup):
        """Observations for policy group with features extracted from RGB images with a frozen Theia-Tiny Transformer"""

        image: Any = field(
            default_factory=lambda: ObsTerm(
                func=mdp.image_features,
                params={
                    "sensor_cfg": SceneEntityCfg("tiled_camera"),
                    "data_type": "rgb",
                    "model_name": "theia-tiny-patch16-224-cddsv",
                    "model_device": "cuda:0",
                },
            )
        )

    policy: ObsGroup = field(default_factory=TheiaTinyFeaturesCameraPolicyCfg)


##
# Environment configuration
##


@dataclass
class CartpoleCameraEnvCfg(PresetCfg):
    """Cartpole environment with a selectable camera observation pipeline.

    A single ``presets=`` selector cascades through this preset: it picks the observation pipeline
    here and, via :attr:`CartpoleCameraSceneCfg.tiled_camera`, the matching camera data type and
    rendering backend. The feature-extractor variants (``resnet18``, ``theia_tiny``) operate on RGB
    images, so they fall back to the default camera.
    """

    @dataclass
    class BaseCartpoleCameraEnvCfg(CartpoleEnvCfg):
        """Camera variant of :class:`CartpoleEnvCfg` -- only the fields that differ are overridden."""

        frame_stack: int = 2
        """Number of frames to stack along the channel dimension.

        Values less than two disable stacking.
        """

        # scene: fewer, more-spaced envs so each camera renders cleanly
        scene: CartpoleCameraSceneCfg = field(
            default_factory=lambda: CartpoleCameraSceneCfg(num_envs=512, env_spacing=20.0)
        )

        def __post_init__(self):
            if parent_post_init := getattr(super(), "__post_init__", None):
                parent_post_init()
            # remove ground as it obstructs the camera
            self.scene.ground = None
            self.events.reset_pole_position.params["position_range"] = (-0.125 * math.pi, 0.125 * math.pi)
            # visualizer camera settings
            self.sim.default_visualizer_cfg = VisualizerCfg(eye=(20.0, 20.0, 20.0), lookat=(0.0, 0.0, 0.0))

    rgb: Any = field(
        default_factory=lambda: CartpoleCameraEnvCfg.BaseCartpoleCameraEnvCfg(
            observations=image_observations_cfg("rgb")
        )
    )
    depth: Any = field(
        default_factory=lambda: CartpoleCameraEnvCfg.BaseCartpoleCameraEnvCfg(
            observations=image_observations_cfg("depth")
        )
    )
    albedo: Any = field(
        default_factory=lambda: CartpoleCameraEnvCfg.BaseCartpoleCameraEnvCfg(
            observations=image_observations_cfg("albedo")
        )
    )
    semantic_segmentation: Any = field(
        default_factory=lambda: CartpoleCameraEnvCfg.BaseCartpoleCameraEnvCfg(
            observations=image_observations_cfg("semantic_segmentation")
        )
    )
    simple_shading_constant_diffuse: Any = field(
        default_factory=lambda: CartpoleCameraEnvCfg.BaseCartpoleCameraEnvCfg(
            observations=image_observations_cfg("simple_shading_constant_diffuse")
        )
    )
    simple_shading_diffuse_mdl: Any = field(
        default_factory=lambda: CartpoleCameraEnvCfg.BaseCartpoleCameraEnvCfg(
            observations=image_observations_cfg("simple_shading_diffuse_mdl")
        )
    )
    simple_shading_full_mdl: Any = field(
        default_factory=lambda: CartpoleCameraEnvCfg.BaseCartpoleCameraEnvCfg(
            observations=image_observations_cfg("simple_shading_full_mdl")
        )
    )
    resnet18: Any = field(
        default_factory=lambda: CartpoleCameraEnvCfg.BaseCartpoleCameraEnvCfg(observations=ResNet18ObservationCfg())
    )
    theia_tiny: Any = field(
        default_factory=lambda: CartpoleCameraEnvCfg.BaseCartpoleCameraEnvCfg(observations=TheiaTinyObservationCfg())
    )
    default: Any = field(
        default_factory=lambda: CartpoleCameraEnvCfg.BaseCartpoleCameraEnvCfg(
            observations=image_observations_cfg("rgb")
        )
    )
