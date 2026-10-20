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

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pxr import Usd

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
    **kwargs: Any,
) -> Usd.Prim:
    """Spawn the Kitchen Set with its own lights deactivated.

    The set ships five lights; cloned into every environment they would multiply with the
    environment count, so they are deactivated in every copy.
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


_IMAGE_OBSERVATION_FUNCS = {
    "rgb": mdp.image_rgb,
    "depth": mdp.image_depth,
    "albedo": mdp.image_rgb,
    "simple_shading_constant_diffuse": mdp.image_rgb,
    "simple_shading_diffuse_mdl": mdp.image_rgb,
    "simple_shading_full_mdl": mdp.image_rgb,
    "semantic_segmentation": mdp.image_segmentation,
}
"""Image observation function per camera data type."""

_IMAGE_RESOLUTIONS = (64, 128, 256)


def _camera_presets(make_option: Callable[[str, int], Any]) -> PresetCfg:
    """One option per data type and resolution, named like the Kuka-Allegro camera presets (e.g. ``rgb64``).

    The camera and its image observation share these names, so one ``presets=`` name selects both.
    """
    options = {
        f"{data_type}{resolution}": make_option(data_type, resolution)
        for data_type in _IMAGE_OBSERVATION_FUNCS
        for resolution in _IMAGE_RESOLUTIONS
    }
    return preset(default=options["rgb64"], **options)


def _kitchen_camera(data_type: str, resolution: int) -> CameraCfg:
    return replace(KITCHEN_CAMERA_CFG, data_types=[data_type], width=resolution, height=resolution)


def _camera_image_term(data_type: str, resolution: int) -> ObsTerm:
    return ObsTerm(
        func=_IMAGE_OBSERVATION_FUNCS[data_type],
        params={
            "sensor_cfg": SceneEntityCfg("camera"),
            "data_type": data_type,
            "normalize": False,
            "channel_first": True,
        },
    )


##
# Environment configuration
##


@configclass
class FrankaKitchenSceneCfg(ReachSceneCfg):
    """Reach scene plus a static, visual-only Kitchen Set and one camera per environment."""

    kitchen: AssetBaseCfg = KITCHEN_CFG
    camera: CameraCfg = _camera_presets(_kitchen_camera)


@configclass
class FrankaKitchenObservationsCfg(ObservationsCfg):
    """Reach observations plus the camera image.

    Reading the image once per step is what triggers the single lazy render per ``env.step``.
    """

    @configclass
    class CameraImageCfg(ObsGroup):
        """Camera observations."""

        image = _camera_presets(_camera_image_term)

    camera_image: CameraImageCfg = CameraImageCfg()


@configclass
class FrankaKitchenCameraEnvCfg(FrankaReachEnvCfg):
    """Franka Reach in a kitchen with one camera per environment."""

    # The kitchen spans y -2.24..2.25 m and the camera sits at y -7.6 m, so the neighbouring kitchen
    # along -Y must end behind the camera (-spacing + 2.25 < -7.6, i.e. spacing > 9.85 m).
    scene: FrankaKitchenSceneCfg = FrankaKitchenSceneCfg(num_envs=256, env_spacing=10.0)
    observations: FrankaKitchenObservationsCfg = FrankaKitchenObservationsCfg()
