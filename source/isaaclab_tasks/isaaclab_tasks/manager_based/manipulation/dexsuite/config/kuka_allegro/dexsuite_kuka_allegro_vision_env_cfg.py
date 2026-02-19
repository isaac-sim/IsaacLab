# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dexsuite Kuka Allegro vision env config.

Tasks: Single-camera tasks (e.g. Isaac-Dexsuite-Kuka-Allegro-Lift-Single-Camera-v0) use
KukaAllegroSingleCameraMixinCfg and single_camera_variants. Duo-camera uses
KukaAllegroDuoCameraMixinCfg and duo_camera_variants; only single-camera tasks are
currently registered (see this package's __init__.py). The task name selects single vs duo;
env.scene then picks resolution/renderer/camera type for that task.

Scene variant convention (local / self-learning use):

  env.scene = "<width>x<height><renderer_tag>_<camera_tag>"

  - width, height: resolution (e.g. 64, 128, 256).
  - renderer_tag: "tiled" → RTX rendering, "newton" → Newton Warp rendering.
  - camera_tag: "rgb" | "depth" | "albedo" (maps to rgb, distance_to_image_plane, diffuse_albedo).

  Examples: 64x64tiled_rgb, 128x128newton_depth. For tests without Isaac Sim, use
  scene_variant_keys.parse_scene_key() and scene_variant_keys.get_scene_variant_keys().
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ... import dexsuite_env_cfg as dexsuite_state_impl
from ... import mdp
from . import dexsuite_kuka_allegro_env_cfg as kuka_allegro_dexsuite
from . import scene_variant_keys as _svk


def _scene_cfg_from_parsed(parsed: dict, scene_cls: type, base: dict) -> object:
    """Build a scene config from parsed scene key and base kwargs."""
    return scene_cls(**{**base, **parsed})


@configclass
class KukaAllegroSingleTiledCameraSceneCfg(kuka_allegro_dexsuite.KukaAllegroSceneCfg):
    """Dexsuite scene for multi-objects Lifting/Reorientation"""

    camera_type: str = "rgb"
    width: int = 64
    height: int = 64
    renderer_type: str = "rtx"  # "rtx" for RTX rendering, "newton_warp" for Warp ray tracing

    base_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.57, -0.8, 0.5),
            rot=(0.6124, 0.3536, 0.3536, 0.6124),
            convention="opengl",
        ),
        data_types=MISSING,
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.01, 2.5)),
        width=MISSING,
        height=MISSING,
        renderer_type=MISSING,
    )

    def __post_init__(self):
        super().__post_init__()
        self.base_camera.data_types = [self.camera_type]
        self.base_camera.width = self.width
        self.base_camera.height = self.height
        # Set renderer type: "rtx" means None (default RTX), "newton_warp" passes through
        self.base_camera.renderer_type = None if self.renderer_type == "rtx" else self.renderer_type
        del self.camera_type
        del self.width
        del self.height
        del self.renderer_type


@configclass
class KukaAllegroDuoTiledCameraSceneCfg(KukaAllegroSingleTiledCameraSceneCfg):
    """Dexsuite scene for multi-objects Lifting/Reorientation"""

    wrist_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/ee_link/palm_link/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.038, -0.38, -0.18),
            rot=(0.641, 0.641, -0.299, 0.299),
            convention="opengl",
        ),
        data_types=MISSING,
        spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.01, 2.5)),
        width=MISSING,
        height=MISSING,
        renderer_type=MISSING,
    )

    def __post_init__(self):
        super().__post_init__()
        self.wrist_camera.data_types = self.base_camera.data_types
        self.wrist_camera.width = self.base_camera.width
        self.wrist_camera.height = self.base_camera.height
        self.wrist_camera.renderer_type = self.base_camera.renderer_type


@configclass
class KukaAllegroSingleCameraObservationsCfg(kuka_allegro_dexsuite.KukaAllegroObservationCfg):
    """Observation specifications for the MDP."""

    @configclass
    class BaseImageObsCfg(ObsGroup):
        """Camera observations for policy group."""

        object_observation_b = ObsTerm(
            func=mdp.vision_camera,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-1.0, 1.0),
            params={"sensor_cfg": SceneEntityCfg("base_camera")},
        )

    base_image: BaseImageObsCfg = BaseImageObsCfg()

    def __post_init__(self):
        super().__post_init__()
        for group in self.__dataclass_fields__.values():
            obs_group = getattr(self, group.name)
            obs_group.history_length = None


@configclass
class KukaAllegroDuoCameraObservationsCfg(KukaAllegroSingleCameraObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class WristImageObsCfg(ObsGroup):
        wrist_observation = ObsTerm(
            func=mdp.vision_camera,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-1.0, 1.0),
            params={"sensor_cfg": SceneEntityCfg("wrist_camera")},
        )

    wrist_image: WristImageObsCfg = WristImageObsCfg()


sa = {"num_envs": 4096, "env_spacing": 3, "replicate_physics": False}


def _build_scene_variants(scene_cls: type) -> dict:
    """Build scene variants from convention: key = <width>x<height><renderer_tag>_<camera_tag>."""
    out = {}
    for (w, h) in _svk.RESOLUTIONS:
        for renderer_tag, camera_tag in _svk.RENDERER_CAMERA_COMBO:
            key = f"{w}x{h}{renderer_tag}_{camera_tag}"
            parsed = {
                "width": w,
                "height": h,
                "renderer_type": _svk.RENDERER_TAG_TO_TYPE[renderer_tag],
                "camera_type": _svk.CAMERA_TAG_TO_TYPE[camera_tag],
            }
            out[key] = _scene_cfg_from_parsed(parsed, scene_cls, sa)
    return out


# Re-export for callers that import from this module (e.g. tests using full env)
parse_scene_key = _svk.parse_scene_key


single_camera_variants = _build_scene_variants(KukaAllegroSingleTiledCameraSceneCfg)
duo_camera_variants = _build_scene_variants(KukaAllegroDuoTiledCameraSceneCfg)


@configclass
class KukaAllegroSingleCameraMixinCfg(kuka_allegro_dexsuite.KukaAllegroMixinCfg):
    scene = KukaAllegroSingleTiledCameraSceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
        camera_type="rgb",
        width=64,
        height=64,
        renderer_type="rtx",
    )
    observations: KukaAllegroSingleCameraObservationsCfg = KukaAllegroSingleCameraObservationsCfg()
    variants: dict = {}

    def __post_init__(self: kuka_allegro_dexsuite.DexsuiteKukaAllegroLiftEnvCfg):
        super().__post_init__()
        self.variants.setdefault("scene", {}).update(single_camera_variants)


@configclass
class KukaAllegroDuoCameraMixinCfg(kuka_allegro_dexsuite.KukaAllegroMixinCfg):
    scene = KukaAllegroDuoTiledCameraSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)
    observations: KukaAllegroDuoCameraObservationsCfg = KukaAllegroDuoCameraObservationsCfg()
    variants: dict = {}

    def __post_init__(self: kuka_allegro_dexsuite.DexsuiteKukaAllegroLiftEnvCfg):
        super().__post_init__()
        self.variants.setdefault("scene", {}).update(duo_camera_variants)


# SingleCamera
@configclass
class DexsuiteKukaAllegroLiftSingleCameraEnvCfg(
    KukaAllegroSingleCameraMixinCfg, dexsuite_state_impl.DexsuiteLiftEnvCfg
):
    pass


@configclass
class DexsuiteKukaAllegroLiftSingleCameraEnvCfg_PLAY(
    KukaAllegroSingleCameraMixinCfg,
    dexsuite_state_impl.DexsuiteLiftEnvCfg_PLAY,
):
    pass


@configclass
class DexsuiteKukaAllegroReorientSingleCameraEnvCfg(
    KukaAllegroSingleCameraMixinCfg, dexsuite_state_impl.DexsuiteReorientEnvCfg
):
    pass


@configclass
class DexsuiteKukaAllegroReorientSingleCameraEnvCfg_PLAY(
    KukaAllegroSingleCameraMixinCfg,
    dexsuite_state_impl.DexsuiteReorientEnvCfg_PLAY,
):
    pass


# DuoCamera
@configclass
class DexsuiteKukaAllegroLiftDuoCameraEnvCfg(KukaAllegroDuoCameraMixinCfg, dexsuite_state_impl.DexsuiteLiftEnvCfg):
    pass


# @configclass
# class DexsuiteKukaAllegroLiftDuoCameraEnvCfg_PLAY(
#     KukaAllegroDuoCameraMixinCfg,
#     dexsuite_state_impl.DexsuiteLiftEnvCfg_PLAY
# ):
#     pass
