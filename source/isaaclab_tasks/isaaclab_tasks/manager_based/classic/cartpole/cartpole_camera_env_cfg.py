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

import isaaclab_tasks.manager_based.classic.cartpole.mdp as mdp
from isaaclab_tasks.utils import PresetCfg

from .cartpole_env_cfg import CartpoleEnvCfg, CartpoleSceneCfg

##
# Scene definition (per-variant, retired Style A -- see deprecation notice below)
##


@configclass
class CartpoleRGBCameraSceneCfg(CartpoleSceneCfg):
    """Configuration for the cartpole environment with RGB camera.

    **Deprecated** -- backs the retired :obj:`Isaac-Cartpole-RGB-v0` task ID
    via :class:`CartpoleRGBCameraEnvCfg`. Use the consolidated
    :class:`CartpoleCameraSceneCfg` (below) for new code. Removed alongside
    the retired task ID.
    """

    # add camera to the scene
    tiled_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=CameraCfg.OffsetCfg(pos=(-7.0, 0.0, 3.0), rot=(0.0, 0.1045, 0.0, 0.9945), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=100,
        height=100,
    )


@configclass
class CartpoleDepthCameraSceneCfg(CartpoleSceneCfg):
    """**Deprecated** -- backs :obj:`Isaac-Cartpole-Depth-v0`. Use
    :class:`CartpoleCameraSceneCfg`."""

    # add camera to the scene
    tiled_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=CameraCfg.OffsetCfg(pos=(-7.0, 0.0, 3.0), rot=(0.0, 0.1045, 0.0, 0.9945), convention="world"),
        data_types=["distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=100,
        height=100,
    )


##
# MDP settings -- observation pipelines (shared by retired Style A subclasses
# and the new ``CartpoleObservationsCfg`` PresetCfg below)
##


@configclass
class RGBObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        image = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: ObsGroup = RGBCameraPolicyCfg()


@configclass
class DepthObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class DepthCameraPolicyCfg(ObsGroup):
        """Observations for policy group with depth images."""

        image = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "distance_to_camera"}
        )

    policy: ObsGroup = DepthCameraPolicyCfg()


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
# Per-variant env configuration (retired Style A -- see deprecation notice below)
#
# These subclasses back the retired :obj:`Isaac-Cartpole-{RGB,Depth,RGB-ResNet18,RGB-TheiaTiny}-v0`
# gym task IDs. The deprecation shims in the sibling ``__init__.py`` route to
# them via ``cfg_factory`` so retired task IDs stay bit-for-bit identical to
# develop. Use the consolidated :class:`CartpoleCameraPresetsEnvCfg` (below)
# for new code; these will be removed alongside the retired task IDs.
##


@configclass
class CartpoleRGBCameraEnvCfg(CartpoleEnvCfg):
    """**Deprecated** -- backs :obj:`Isaac-Cartpole-RGB-v0`. Migration:
    ``--task=Isaac-Cartpole-Camera-v0 presets=rgb``."""

    scene: CartpoleRGBCameraSceneCfg = CartpoleRGBCameraSceneCfg(num_envs=512, env_spacing=20)
    observations: RGBObservationsCfg = RGBObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # remove ground as it obstructs the camera
        self.scene.ground = None
        # viewer settings
        self.viewer.eye = (7.0, 0.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 2.5)


@configclass
class CartpoleDepthCameraEnvCfg(CartpoleEnvCfg):
    """**Deprecated** -- backs :obj:`Isaac-Cartpole-Depth-v0`. Migration:
    ``--task=Isaac-Cartpole-Camera-v0 presets=depth``."""

    scene: CartpoleDepthCameraSceneCfg = CartpoleDepthCameraSceneCfg(num_envs=512, env_spacing=20)
    observations: DepthObservationsCfg = DepthObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # remove ground as it obstructs the camera
        self.scene.ground = None
        # viewer settings
        self.viewer.eye = (7.0, 0.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 2.5)


@configclass
class CartpoleResNet18CameraEnvCfg(CartpoleRGBCameraEnvCfg):
    """**Deprecated** -- backs :obj:`Isaac-Cartpole-RGB-ResNet18-v0`. Migration:
    ``--task=Isaac-Cartpole-Camera-v0 --agent=rl_games_feature_cfg_entry_point presets=resnet18``.
    """

    observations: ResNet18ObservationCfg = ResNet18ObservationCfg()


@configclass
class CartpoleTheiaTinyCameraEnvCfg(CartpoleRGBCameraEnvCfg):
    """**Deprecated** -- backs :obj:`Isaac-Cartpole-RGB-TheiaTiny-v0`. Migration:
    ``--task=Isaac-Cartpole-Camera-v0 --agent=rl_games_feature_cfg_entry_point presets=theia_tiny``.
    """

    observations: TheiaTinyObservationCfg = TheiaTinyObservationCfg()


##
# Consolidated env configuration (canonical -- used by Isaac-Cartpole-Camera-v0)
##


@configclass
class CartpoleCameraDataTypesCfg(PresetCfg):
    """Camera ``data_types`` selector for the manager-based cartpole camera.

    The camera pose, intrinsics, and resolution are identical for every
    variant; only the data-type stream changes (``rgb`` for the RGB / feature
    pipelines, ``distance_to_camera`` for depth). Keeping just the data-type
    list in a small PresetCfg lets the parent scene cfg share a single camera.
    """

    rgb = ["rgb"]
    resnet18 = rgb
    theia_tiny = rgb
    depth = ["distance_to_camera"]
    default = rgb


@configclass
class CartpoleCameraSceneCfg(CartpoleSceneCfg):
    """Scene cfg with a single tiled camera whose data-type stream varies per preset."""

    tiled_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=CameraCfg.OffsetCfg(pos=(-7.0, 0.0, 3.0), rot=(0.0, 0.1045, 0.0, 0.9945), convention="world"),
        data_types=CartpoleCameraDataTypesCfg(),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=100,
        height=100,
    )


@configclass
class CartpoleObservationsCfg(PresetCfg):
    """Observation-pipeline selector for the manager-based cartpole camera task."""

    rgb: RGBObservationsCfg = RGBObservationsCfg()
    depth: DepthObservationsCfg = DepthObservationsCfg()
    resnet18: ResNet18ObservationCfg = ResNet18ObservationCfg()
    theia_tiny: TheiaTinyObservationCfg = TheiaTinyObservationCfg()
    default = rgb


@configclass
class CartpoleCameraPresetsEnvCfg(CartpoleEnvCfg):
    """Manager-based cartpole perception with selectable observation pipeline.

    Variants selected via ``presets=<name>``:

    * ``rgb`` / ``default`` -- raw RGB camera observations.
    * ``depth`` -- depth (distance-to-camera) observations.
    * ``resnet18`` -- features extracted by a frozen ResNet18 backbone from
      the RGB camera.
    * ``theia_tiny`` -- features extracted by a frozen Theia-Tiny transformer
      backbone from the RGB camera.

    The varying parts (``scene.tiled_camera.data_types`` and ``observations``)
    are nested :class:`~isaaclab_tasks.utils.PresetCfg` fields. The framework
    resolver pins both at ``gym.make`` time when the user passes
    ``presets=<name>``; all other fields are inherited unchanged from
    :class:`CartpoleEnvCfg` (including the ``physics=`` typed selector wired
    through ``CartpolePhysicsCfg``).

    Used by the canonical :obj:`Isaac-Cartpole-Camera-v0` task. The retired
    per-variant task IDs (:obj:`Isaac-Cartpole-{RGB,Depth,RGB-ResNet18,RGB-TheiaTiny}-v0`)
    return the legacy per-variant subclasses above instead, via the
    deprecation shims in the sibling ``__init__.py``.
    """

    scene: CartpoleCameraSceneCfg = CartpoleCameraSceneCfg(num_envs=512, env_spacing=20)
    observations: CartpoleObservationsCfg = CartpoleObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # remove ground as it obstructs the camera
        self.scene.ground = None
        # viewer settings
        self.viewer.eye = (7.0, 0.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 2.5)
