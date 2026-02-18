# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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

# RTX rendering variants
singe_camera_variants = {
    "64x64tiled_depth": KukaAllegroSingleTiledCameraSceneCfg(
        **{**sa, "camera_type": "distance_to_image_plane", "width": 64, "height": 64, "renderer_type": "rtx"}
    ),
    "64x64tiled_rgb": KukaAllegroSingleTiledCameraSceneCfg(
        **{**sa, "camera_type": "rgb", "width": 64, "height": 64, "renderer_type": "rtx"}
    ),
    "64x64tiled_albedo": KukaAllegroSingleTiledCameraSceneCfg(
        **{**sa, "camera_type": "diffuse_albedo", "width": 64, "height": 64, "renderer_type": "rtx"}
    ),
    "128x128tiled_depth": KukaAllegroSingleTiledCameraSceneCfg(
        **{**sa, "camera_type": "distance_to_image_plane", "width": 128, "height": 128, "renderer_type": "rtx"}
    ),
    "128x128tiled_rgb": KukaAllegroSingleTiledCameraSceneCfg(
        **{**sa, "camera_type": "rgb", "width": 128, "height": 128, "renderer_type": "rtx"}
    ),
    "128x128tiled_albedo": KukaAllegroSingleTiledCameraSceneCfg(
        **{**sa, "camera_type": "diffuse_albedo", "width": 128, "height": 128, "renderer_type": "rtx"}
    ),
    "256x256tiled_depth": KukaAllegroSingleTiledCameraSceneCfg(
        **{**sa, "camera_type": "distance_to_image_plane", "width": 256, "height": 256, "renderer_type": "rtx"}
    ),
    "256x256tiled_rgb": KukaAllegroSingleTiledCameraSceneCfg(
        **{**sa, "camera_type": "rgb", "width": 256, "height": 256, "renderer_type": "rtx"}
    ),
    "256x256tiled_albedo": KukaAllegroSingleTiledCameraSceneCfg(
        **{**sa, "camera_type": "diffuse_albedo", "width": 256, "height": 256, "renderer_type": "rtx"}
    ),
}
duo_camera_variants = {
    "64x64tiled_depth": KukaAllegroDuoTiledCameraSceneCfg(
        **{**sa, "camera_type": "distance_to_image_plane", "width": 64, "height": 64, "renderer_type": "rtx"}
    ),
    "64x64tiled_rgb": KukaAllegroDuoTiledCameraSceneCfg(
        **{**sa, "camera_type": "rgb", "width": 64, "height": 64, "renderer_type": "rtx"}
    ),
    "64x64tiled_albedo": KukaAllegroDuoTiledCameraSceneCfg(
        **{**sa, "camera_type": "diffuse_albedo", "width": 64, "height": 64, "renderer_type": "rtx"}
    ),
    "128x128tiled_depth": KukaAllegroDuoTiledCameraSceneCfg(
        **{**sa, "camera_type": "distance_to_image_plane", "width": 128, "height": 128, "renderer_type": "rtx"}
    ),
    "128x128tiled_rgb": KukaAllegroDuoTiledCameraSceneCfg(
        **{**sa, "camera_type": "rgb", "width": 128, "height": 128, "renderer_type": "rtx"}
    ),
    "128x128tiled_albedo": KukaAllegroDuoTiledCameraSceneCfg(
        **{**sa, "camera_type": "diffuse_albedo", "width": 128, "height": 128, "renderer_type": "rtx"}
    ),
    "256x256tiled_depth": KukaAllegroDuoTiledCameraSceneCfg(
        **{**sa, "camera_type": "distance_to_image_plane", "width": 256, "height": 256, "renderer_type": "rtx"}
    ),
    "256x256tiled_rgb": KukaAllegroDuoTiledCameraSceneCfg(
        **{**sa, "camera_type": "rgb", "width": 256, "height": 256, "renderer_type": "rtx"}
    ),
    "256x256tiled_albedo": KukaAllegroDuoTiledCameraSceneCfg(
        **{**sa, "camera_type": "diffuse_albedo", "width": 256, "height": 256, "renderer_type": "rtx"}
    ),
}

# Newton Warp rendering variants
single_camera_newton_warp_variants = {
    "64x64newton_depth": KukaAllegroSingleTiledCameraSceneCfg(
        **{**sa, "camera_type": "distance_to_image_plane", "width": 64, "height": 64, "renderer_type": "newton_warp"}
    ),
    "64x64newton_rgb": KukaAllegroSingleTiledCameraSceneCfg(
        **{**sa, "camera_type": "rgb", "width": 64, "height": 64, "renderer_type": "newton_warp"}
    ),
    "128x128newton_depth": KukaAllegroSingleTiledCameraSceneCfg(
        **{**sa, "camera_type": "distance_to_image_plane", "width": 128, "height": 128, "renderer_type": "newton_warp"}
    ),
    "128x128newton_rgb": KukaAllegroSingleTiledCameraSceneCfg(
        **{**sa, "camera_type": "rgb", "width": 128, "height": 128, "renderer_type": "newton_warp"}
    ),
    "256x256newton_depth": KukaAllegroSingleTiledCameraSceneCfg(
        **{**sa, "camera_type": "distance_to_image_plane", "width": 256, "height": 256, "renderer_type": "newton_warp"}
    ),
    "256x256newton_rgb": KukaAllegroSingleTiledCameraSceneCfg(
        **{**sa, "camera_type": "rgb", "width": 256, "height": 256, "renderer_type": "newton_warp"}
    ),
}


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
        self.variants.setdefault("scene", {}).update(singe_camera_variants)
        self.variants["scene"].update(single_camera_newton_warp_variants)


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


# @configclass
# class DexsuiteKukaAllegroLiftSingleCameraEnvCfg_PLAY(
#     KukaAllegroSingleCameraMixinCfg,
#     dexsuite_state_impl.DexsuiteLiftEnvCfg_PLAY
# ):
#     pass


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
