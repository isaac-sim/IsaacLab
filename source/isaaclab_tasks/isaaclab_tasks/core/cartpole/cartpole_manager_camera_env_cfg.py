# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the manager-based cartpole camera environment."""

import math

from isaaclab_ov.renderers import OVRTXRendererCfg

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, LidarCfg
from isaaclab.utils import configclass
from isaaclab.utils.images import is_depth_like
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


def _cartpole_lidar_pattern() -> tuple[list[float], list[float], list[int], list[int]]:
    """Build a 7 by 7 solid-state fan aimed at the cart.

    The cart is about 20 cm across and the pole is a few centimeters wide. A sensor one meter
    away, at the cart's height, hits that body with a two-degree grid. A lidar several meters
    away looks past it.
    """
    azimuths = [float(value) for value in range(-6, 7, 2)]
    elevations = [float(value) for value in range(-6, 7, 2)]
    azimuth_deg: list[float] = []
    elevation_deg: list[float] = []
    fire_time_ns: list[int] = []
    channel_id: list[int] = []
    index = 0
    for elevation in elevations:
        for azimuth in azimuths:
            azimuth_deg.append(azimuth)
            elevation_deg.append(elevation)
            fire_time_ns.append(index * 1000)
            channel_id.append(index + 1)
            index += 1
    return azimuth_deg, elevation_deg, fire_time_ns, channel_id


_CARTPOLE_LIDAR_AZIMUTH, _CARTPOLE_LIDAR_ELEVATION, _CARTPOLE_LIDAR_TIME, _CARTPOLE_LIDAR_CHANNEL = (
    _cartpole_lidar_pattern()
)


@configclass
class CartpoleLidarSceneCfg(CartpoleCameraSceneCfg):
    """Side camera plus one solid-state lidar, both captured by the same OVRTX renderer."""

    # The cartpole camera presets keep a selectable renderer. This concrete camera is pinned so
    # the lidar preset does not fall through to the Newton Warp renderer.
    tiled_camera: CameraCfg = CartpoleTiledCameraCfg.BaseCartpoleTiledCameraCfg(
        data_types=["rgb"],
        renderer_cfg=OVRTXRendererCfg(motion_bvh="auto"),
    )
    lidar: LidarCfg = LidarCfg(
        prim_path="{ENV_REGEX_NS}/Lidar",
        # Same orientation as the OVRTX lidar fixture (rotateXYZ 90, 0, -90), looking along world +X.
        offset=LidarCfg.OffsetCfg(pos=(-1.0, 0.0, 2.0), rot=(0.5, -0.5, -0.5, 0.5)),
        azimuth_deg=_CARTPOLE_LIDAR_AZIMUTH,
        elevation_deg=_CARTPOLE_LIDAR_ELEVATION,
        fire_time_ns=_CARTPOLE_LIDAR_TIME,
        channel_id=_CARTPOLE_LIDAR_CHANNEL,
        renderer_cfg=OVRTXRendererCfg(motion_bvh="auto"),
    )


##
# MDP settings
##


def image_observations_cfg(data_type: str):
    """Build a single-camera-image policy observation group.

    The policy image is channel-first and stacks the last two frames; set the term's
    ``frame_stack`` parameter to change the stack size.

    Args:
        data_type: Camera data type to read from the tiled camera (e.g. ``"rgb"``, ``"depth"``).

    Returns:
        An observations config with camera policy observations and privileged state critic observations.
    """
    if is_depth_like(data_type):
        image_term = mdp.image_depth
    elif "segmentation" in data_type:
        image_term = mdp.image_segmentation
    else:
        image_term = mdp.image_rgb

    @configclass
    class ImageObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            image = ObsTerm(
                func=image_term,
                params={
                    "sensor_cfg": SceneEntityCfg("tiled_camera"),
                    "data_type": data_type,
                    "channel_first": True,
                    "frame_stack": 2,
                },
            )

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        policy: ObsGroup = PolicyCfg()
        critic: ObsGroup = ObservationsCfg.PolicyCfg()

    return ImageObservationsCfg()


def lidar_observations_cfg():
    """Build policy observations from the cartpole joints and the lidar point cloud.

    The critic keeps the privileged joint state. Missed rays are zeros in the policy term and
    stay non-finite on the sensor itself.

    Returns:
        An observations config with the joint-and-lidar policy and the privileged critic.
    """

    @configclass
    class LidarObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
            joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
            lidar_points = ObsTerm(func=mdp.lidar_points, params={"sensor_cfg": SceneEntityCfg("lidar")})

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        policy: ObsGroup = PolicyCfg()
        critic: ObsGroup = ObservationsCfg.PolicyCfg()

    return LidarObservationsCfg()


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
    images, so they fall back to the default camera. The ``lidar`` variant keeps the side RGB camera
    and adds a tiled OVRTX lidar; both are pinned to OVRTX with motion BVH enabled.
    """

    @configclass
    class BaseCartpoleCameraEnvCfg(CartpoleEnvCfg):
        """Camera variant of :class:`CartpoleEnvCfg`; only the fields that differ are overridden."""

        # scene: fewer, more-spaced envs so each camera renders cleanly
        scene: CartpoleCameraSceneCfg = CartpoleCameraSceneCfg(num_envs=512, env_spacing=20.0)

        def __post_init__(self):
            super().__post_init__()
            # remove the ground as it obstructs the camera
            self.scene.ground = None
            # reset: smaller initial pole angle than the proprioceptive task
            self.events.reset_pole_position.params["position_range"] = (-0.125 * math.pi, 0.125 * math.pi)
            # visualizer settings
            self.sim.visualizer_cfgs = VisualizerCfg(eye=(20.0, 20.0, 20.0), lookat=(0.0, 0.0, 0.0))

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
    lidar = BaseCartpoleCameraEnvCfg(
        scene=CartpoleLidarSceneCfg(num_envs=512, env_spacing=20.0),
        observations=lidar_observations_cfg(),
    )
    default = rgb
