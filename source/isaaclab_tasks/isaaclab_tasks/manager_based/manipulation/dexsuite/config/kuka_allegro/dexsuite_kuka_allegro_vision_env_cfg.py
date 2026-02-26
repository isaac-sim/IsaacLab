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

  Neutral key + override (pass env.scene=... before the override):
    env.scene = "<width>x<height><camera_tag>"   e.g. 64x64rgb, 64x64depth
    env.scene.base_camera.renderer_type = rtx | warp_renderer
    Default renderer for neutral keys is rtx. train.py sorts CLI so env.scene= comes first.

  Workflow: Hydra merges env.scene.base_camera.renderer_type=... into the composed config;
  train.py calls env_cfg.from_dict(hydra_env_cfg["env"]), which sets scene.base_camera.renderer_type
  (e.g. "warp_renderer"). We do *not* set base_camera.renderer_cfg in the scene so validation
  and Hydra override work. TiledCamera resolves the string to a renderer at runtime:
  renderer_type "warp_renderer" -> NewtonWarpRendererCfg().create_renderer(), "rtx"/None -> RTX path.
"""

from dataclasses import MISSING, fields

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
    renderer_type: str = "rtx"  # Hydra: env.scene.base_camera.renderer_type=warp_renderer or =rtx

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
        renderer_type=None,  # set in __post_init__ from scene; Hydra may override via env.scene.base_camera.renderer_type=
    )

    def __post_init__(self):
        super().__post_init__()
        self.base_camera.data_types = [self.camera_type]
        self.base_camera.width = self.width
        self.base_camera.height = self.height
        # renderer_type; Hydra may override via env.scene.base_camera.renderer_type=.
        renderer_type_str = getattr(self.base_camera, "renderer_type", None) or self.renderer_type
        if renderer_type_str in ("warp_renderer", "newton_warp"):
            from isaaclab.renderers import renderer_cfg_from_type
            self.base_camera.renderer_cfg = renderer_cfg_from_type(renderer_type_str)
        if hasattr(self.base_camera.renderer_cfg, "data_types"):
            self.base_camera.renderer_cfg.data_types = list(self.base_camera.data_types)
        if hasattr(self.base_camera.renderer_cfg, "width"):
            self.base_camera.renderer_cfg.width = self.base_camera.width
        if hasattr(self.base_camera.renderer_cfg, "height"):
            self.base_camera.renderer_cfg.height = self.base_camera.height
        if hasattr(self.base_camera.renderer_cfg, "num_cameras"):
            self.base_camera.renderer_cfg.num_cameras = 1
        self.base_camera.renderer_type = None if renderer_type_str == "rtx" else renderer_type_str
        # Remove so InteractiveScene._add_entities_from_cfg() does not treat them as assets
        del self.camera_type
        del self.width
        del self.height
        del self.renderer_type

    def __repr__(self):
        # Deleted fields (camera_type, width, height, renderer_type) would break default dataclass __repr__
        parts = []
        for f in fields(self):
            if f.name in ("camera_type", "width", "height", "renderer_type"):
                continue
            try:
                val = getattr(self, f.name)
            except AttributeError:
                continue
            parts.append(f"{f.name}={val!r}")
        return f"{type(self).__name__}({', '.join(parts)})"


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
        renderer_type=None,  # set in __post_init__ from base_camera
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


def _build_neutral_scene_variants(scene_cls: type) -> dict:
    """Build neutral scene variants: key = <width>x<height><camera_tag> (e.g. 64x64rgb, 64x64depth).

    Renderer defaults to rtx; override with env.scene.base_camera.renderer_type=rtx|warp_renderer.
    Pass env.scene=... before env.scene.base_camera.renderer_type=... on the CLI.
    """
    out = {}
    for key in _svk.get_neutral_scene_variant_keys():
        parsed = _svk.parse_neutral_scene_key(key)
        if parsed is None:
            continue
        out[key] = _scene_cfg_from_parsed(parsed, scene_cls, sa)
    return out


# Re-export for callers that import from this module (e.g. tests using full env)
parse_scene_key = _svk.parse_scene_key


single_camera_variants = _build_scene_variants(KukaAllegroSingleTiledCameraSceneCfg)
single_camera_variants.update(_build_neutral_scene_variants(KukaAllegroSingleTiledCameraSceneCfg))
duo_camera_variants = _build_scene_variants(KukaAllegroDuoTiledCameraSceneCfg)
duo_camera_variants.update(_build_neutral_scene_variants(KukaAllegroDuoTiledCameraSceneCfg))


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

    def __repr__(self):
        # Hydra may delete 'variants'; avoid AttributeError in dataclass __repr__
        parts = []
        for f in fields(self):
            try:
                val = getattr(self, f.name)
            except AttributeError:
                continue
            parts.append(f"{f.name}={val!r}")
        return f"{type(self).__name__}({', '.join(parts)})"


@configclass
class KukaAllegroDuoCameraMixinCfg(kuka_allegro_dexsuite.KukaAllegroMixinCfg):
    scene = KukaAllegroDuoTiledCameraSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)
    observations: KukaAllegroDuoCameraObservationsCfg = KukaAllegroDuoCameraObservationsCfg()
    variants: dict = {}

    def __post_init__(self: kuka_allegro_dexsuite.DexsuiteKukaAllegroLiftEnvCfg):
        super().__post_init__()
        self.variants.setdefault("scene", {}).update(duo_camera_variants)

    def __repr__(self):
        # Hydra may delete 'variants'; avoid AttributeError in dataclass __repr__
        parts = []
        for f in fields(self):
            try:
                val = getattr(self, f.name)
            except AttributeError:
                continue
            parts.append(f"{f.name}={val!r}")
        return f"{type(self).__name__}({', '.join(parts)})"


def _safe_env_cfg_repr(self) -> str:
    """Repr that skips missing attributes (e.g. variants deleted by Hydra)."""
    parts = []
    for f in fields(self):
        try:
            val = getattr(self, f.name)
        except AttributeError:
            continue
        parts.append(f"{f.name}={val!r}")
    return f"{type(self).__name__}({', '.join(parts)})"


# SingleCamera
@configclass
class DexsuiteKukaAllegroLiftSingleCameraEnvCfg(
    KukaAllegroSingleCameraMixinCfg, dexsuite_state_impl.DexsuiteLiftEnvCfg
):
    def __repr__(self):
        return _safe_env_cfg_repr(self)


@configclass
class DexsuiteKukaAllegroLiftSingleCameraEnvCfg_PLAY(
    KukaAllegroSingleCameraMixinCfg,
    dexsuite_state_impl.DexsuiteLiftEnvCfg_PLAY,
):
    def __repr__(self):
        return _safe_env_cfg_repr(self)


@configclass
class DexsuiteKukaAllegroReorientSingleCameraEnvCfg(
    KukaAllegroSingleCameraMixinCfg, dexsuite_state_impl.DexsuiteReorientEnvCfg
):
    def __repr__(self):
        return _safe_env_cfg_repr(self)


@configclass
class DexsuiteKukaAllegroReorientSingleCameraEnvCfg_PLAY(
    KukaAllegroSingleCameraMixinCfg,
    dexsuite_state_impl.DexsuiteReorientEnvCfg_PLAY,
):
    def __repr__(self):
        return _safe_env_cfg_repr(self)


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
