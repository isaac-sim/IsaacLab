# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Reach with a static Pixar Kitchen Set and one camera per environment.

The scene is :class:`~isaaclab_tasks.core.reach.config.franka.franka_reach_env_cfg.FrankaReachEnvCfg`
plus the Kitchen Set spawned visual-only under every environment root, so physics sees only the arm and
the table while the renderer sees the whole room. The camera data type and resolution are ``presets=``
selectable with the same names as the Kuka-Allegro camera presets, e.g. ``presets=simple_shading_full_mdl64``.

The Kitchen Set is not part of the Isaac Lab assets, so its path is left missing and must be passed as
``env.scene.kitchen.spawn.usd_path=<path to Kitchen_set.usda>``. Its ``assets`` directory must sit next to it.
"""

from dataclasses import MISSING

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import UsdFileCfg
from isaaclab.utils import configclass, replace

from isaaclab_tasks.core.reach.config.franka.franka_reach_env_cfg import FrankaReachEnvCfg
from isaaclab_tasks.core.reach.reach_env_cfg import ObservationsCfg, ReachSceneCfg
from isaaclab_tasks.utils import PresetCfg, preset
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

##
# Kitchen
##


def spawn_kitchen_set(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the Kitchen Set with its own lights deactivated.

    The set ships five lights; cloned into every environment they would multiply with the
    environment count, so the scene's dome light stays the only light.
    """
    from pxr import Usd, UsdLux  # noqa: PLC0415

    kitchen_prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    for root_prim in sim_utils.find_matching_prims(prim_path, stage=kitchen_prim.GetStage()):
        for prim in Usd.PrimRange(root_prim):
            if prim.HasAPI(UsdLux.LightAPI):
                prim.SetActive(False)
    return kitchen_prim


KITCHEN_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Kitchen",
    # z = -1.05 m is the height of the Reach scene's ground plane.
    init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.71, 0.28, -1.05)),
    spawn=UsdFileCfg(
        func=spawn_kitchen_set,
        usd_path=MISSING,
        # Kitchen_set.usda is authored in centimetres.
        scale=(0.01, 0.01, 0.01),
    ),
)

##
# Camera presets
##

KITCHEN_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Camera",
    # From outside the set's open -Y side looking +Y, below the z ~= 1.47 m ceiling.
    offset=CameraCfg.OffsetCfg(
        pos=(-0.3, -7.6, 0.6),
        rot=(-0.0234, 0.0242, 0.6941, 0.7191),
        convention="world",
    ),
    data_types=MISSING,
    spawn=sim_utils.PinholeCameraCfg(
        # 47 mm puts the room's top edge just above the frame, so no pixel looks past the ceiling.
        focal_length=47.0,
        focus_distance=3.0,
        horizontal_aperture=20.955,
        clipping_range=(0.1, 20.0),
    ),
    width=MISSING,
    height=MISSING,
    renderer_cfg=MultiBackendRendererCfg(),
)


@configclass
class KitchenCameraCfg(PresetCfg):
    """Kitchen camera presets, one per data type and resolution."""

    rgb64 = replace(KITCHEN_CAMERA_CFG, data_types=["rgb"], width=64, height=64)
    rgb128 = replace(KITCHEN_CAMERA_CFG, data_types=["rgb"], width=128, height=128)
    rgb256 = replace(KITCHEN_CAMERA_CFG, data_types=["rgb"], width=256, height=256)
    depth64 = replace(KITCHEN_CAMERA_CFG, data_types=["depth"], width=64, height=64)
    depth128 = replace(KITCHEN_CAMERA_CFG, data_types=["depth"], width=128, height=128)
    depth256 = replace(KITCHEN_CAMERA_CFG, data_types=["depth"], width=256, height=256)
    albedo64 = replace(KITCHEN_CAMERA_CFG, data_types=["albedo"], width=64, height=64)
    albedo128 = replace(KITCHEN_CAMERA_CFG, data_types=["albedo"], width=128, height=128)
    albedo256 = replace(KITCHEN_CAMERA_CFG, data_types=["albedo"], width=256, height=256)
    simple_shading_constant_diffuse64 = replace(
        KITCHEN_CAMERA_CFG, data_types=["simple_shading_constant_diffuse"], width=64, height=64
    )
    simple_shading_constant_diffuse128 = replace(
        KITCHEN_CAMERA_CFG, data_types=["simple_shading_constant_diffuse"], width=128, height=128
    )
    simple_shading_constant_diffuse256 = replace(
        KITCHEN_CAMERA_CFG, data_types=["simple_shading_constant_diffuse"], width=256, height=256
    )
    simple_shading_diffuse_mdl64 = replace(
        KITCHEN_CAMERA_CFG, data_types=["simple_shading_diffuse_mdl"], width=64, height=64
    )
    simple_shading_diffuse_mdl128 = replace(
        KITCHEN_CAMERA_CFG, data_types=["simple_shading_diffuse_mdl"], width=128, height=128
    )
    simple_shading_diffuse_mdl256 = replace(
        KITCHEN_CAMERA_CFG, data_types=["simple_shading_diffuse_mdl"], width=256, height=256
    )
    simple_shading_full_mdl64 = replace(KITCHEN_CAMERA_CFG, data_types=["simple_shading_full_mdl"], width=64, height=64)
    simple_shading_full_mdl128 = replace(
        KITCHEN_CAMERA_CFG, data_types=["simple_shading_full_mdl"], width=128, height=128
    )
    simple_shading_full_mdl256 = replace(
        KITCHEN_CAMERA_CFG, data_types=["simple_shading_full_mdl"], width=256, height=256
    )
    semantic_segmentation64 = replace(KITCHEN_CAMERA_CFG, data_types=["semantic_segmentation"], width=64, height=64)
    semantic_segmentation128 = replace(KITCHEN_CAMERA_CFG, data_types=["semantic_segmentation"], width=128, height=128)
    semantic_segmentation256 = replace(KITCHEN_CAMERA_CFG, data_types=["semantic_segmentation"], width=256, height=256)
    default = rgb64


def _image_term_presets(sensor_name: str) -> PresetCfg:
    """Image observation terms named like the camera presets, so one preset selects both."""
    terms = {}
    for name, func, data_type in (
        ("rgb", mdp.image_rgb, "rgb"),
        ("albedo", mdp.image_rgb, "albedo"),
        ("simple_shading_constant_diffuse", mdp.image_rgb, "simple_shading_constant_diffuse"),
        ("simple_shading_diffuse_mdl", mdp.image_rgb, "simple_shading_diffuse_mdl"),
        ("simple_shading_full_mdl", mdp.image_rgb, "simple_shading_full_mdl"),
        ("semantic_segmentation", mdp.image_segmentation, "semantic_segmentation"),
        ("depth", mdp.image_depth, "depth"),
    ):
        for resolution in (64, 128, 256):
            terms[f"{name}{resolution}"] = ObsTerm(
                func=func,
                params={
                    "sensor_cfg": SceneEntityCfg(sensor_name),
                    "data_type": data_type,
                    "normalize": False,
                    "channel_first": True,
                },
            )
    return preset(default=terms["rgb64"], **terms)


##
# Environment configuration
##


@configclass
class FrankaKitchenSceneCfg(ReachSceneCfg):
    """Reach scene plus a static, visual-only Kitchen Set and one camera per environment."""

    kitchen: AssetBaseCfg = KITCHEN_CFG
    camera: CameraCfg = KitchenCameraCfg()


@configclass
class FrankaKitchenObservationsCfg(ObservationsCfg):
    """Reach observations plus the camera image.

    Reading the image once per step is what triggers the single lazy render per ``env.step``.
    """

    @configclass
    class CameraImageCfg(ObsGroup):
        """Camera observations."""

        image = _image_term_presets("camera")

    camera_image: CameraImageCfg = CameraImageCfg()


@configclass
class FrankaKitchenCameraEnvCfg(FrankaReachEnvCfg):
    """Franka Reach in a kitchen with one camera per environment."""

    # The kitchen spans y -2.24..2.25 m and the camera sits at y -7.6 m, so the neighbouring kitchen
    # along -Y must end behind the camera (-spacing + 2.25 < -7.6, i.e. spacing > 9.85 m).
    scene: FrankaKitchenSceneCfg = FrankaKitchenSceneCfg(num_envs=256, env_spacing=10.0)
    observations: FrankaKitchenObservationsCfg = FrankaKitchenObservationsCfg()
