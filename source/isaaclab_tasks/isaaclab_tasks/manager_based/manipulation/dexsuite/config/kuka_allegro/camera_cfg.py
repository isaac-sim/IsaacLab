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
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.utils import PresetCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from ... import dexsuite_env_cfg as dexsuite
from ... import mdp

FINGERTIP_LIST = ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"]


BASE_CAMERA_CFG = TiledCameraCfg(
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
    renderer_cfg=MultiBackendRendererCfg(),
)

WRIST_CAMERA_CFG = TiledCameraCfg(
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
    renderer_cfg=MultiBackendRendererCfg(),
)


@configclass
class BaseTiledCameraCfg(PresetCfg):
    """Tiled camera configurations"""

    rgb64 = BASE_CAMERA_CFG.replace(data_types=["rgb"], width=64, height=64)
    rgb128 = BASE_CAMERA_CFG.replace(data_types=["rgb"], width=128, height=128)
    rgb256 = BASE_CAMERA_CFG.replace(data_types=["rgb"], width=256, height=256)
    depth64 = BASE_CAMERA_CFG.replace(data_types=["distance_to_image_plane"], width=64, height=64)
    depth128 = BASE_CAMERA_CFG.replace(data_types=["distance_to_image_plane"], width=128, height=128)
    depth256 = BASE_CAMERA_CFG.replace(data_types=["distance_to_image_plane"], width=256, height=256)
    albedo64 = BASE_CAMERA_CFG.replace(data_types=["albedo"], width=64, height=64)
    albedo128 = BASE_CAMERA_CFG.replace(data_types=["albedo"], width=128, height=128)
    albedo256 = BASE_CAMERA_CFG.replace(data_types=["albedo"], width=256, height=256)
    simple_shading_constant_diffuse64 = BASE_CAMERA_CFG.replace(
        data_types=["simple_shading_constant_diffuse"], width=64, height=64
    )
    simple_shading_constant_diffuse128 = BASE_CAMERA_CFG.replace(
        data_types=["simple_shading_constant_diffuse"], width=128, height=128
    )
    simple_shading_constant_diffuse256 = BASE_CAMERA_CFG.replace(
        data_types=["simple_shading_constant_diffuse"], width=256, height=256
    )
    simple_shading_diffuse_mdl64 = BASE_CAMERA_CFG.replace(
        data_types=["simple_shading_diffuse_mdl"], width=64, height=64
    )
    simple_shading_diffuse_mdl128 = BASE_CAMERA_CFG.replace(
        data_types=["simple_shading_diffuse_mdl"], width=128, height=128
    )
    simple_shading_diffuse_mdl256 = BASE_CAMERA_CFG.replace(
        data_types=["simple_shading_diffuse_mdl"], width=256, height=256
    )
    simple_shading_full_mdl64 = BASE_CAMERA_CFG.replace(data_types=["simple_shading_full_mdl"], width=64, height=64)
    simple_shading_full_mdl128 = BASE_CAMERA_CFG.replace(data_types=["simple_shading_full_mdl"], width=128, height=128)
    simple_shading_full_mdl256 = BASE_CAMERA_CFG.replace(data_types=["simple_shading_full_mdl"], width=256, height=256)
    default = rgb64


@configclass
class WristTiledCameraCfg(PresetCfg):
    """Tiled camera configurations"""

    rgb64 = WRIST_CAMERA_CFG.replace(data_types=["rgb"], width=64, height=64)
    rgb128 = WRIST_CAMERA_CFG.replace(data_types=["rgb"], width=128, height=128)
    rgb256 = WRIST_CAMERA_CFG.replace(data_types=["rgb"], width=256, height=256)
    depth64 = WRIST_CAMERA_CFG.replace(data_types=["distance_to_image_plane"], width=64, height=64)
    depth128 = WRIST_CAMERA_CFG.replace(data_types=["distance_to_image_plane"], width=128, height=128)
    depth256 = WRIST_CAMERA_CFG.replace(data_types=["distance_to_image_plane"], width=256, height=256)
    albedo64 = WRIST_CAMERA_CFG.replace(data_types=["albedo"], width=64, height=64)
    albedo128 = WRIST_CAMERA_CFG.replace(data_types=["albedo"], width=128, height=128)
    albedo256 = WRIST_CAMERA_CFG.replace(data_types=["albedo"], width=256, height=256)
    simple_shading_constant_diffuse64 = BASE_CAMERA_CFG.replace(
        data_types=["simple_shading_constant_diffuse"], width=64, height=64
    )
    simple_shading_constant_diffuse128 = BASE_CAMERA_CFG.replace(
        data_types=["simple_shading_constant_diffuse"], width=128, height=128
    )
    simple_shading_constant_diffuse256 = BASE_CAMERA_CFG.replace(
        data_types=["simple_shading_constant_diffuse"], width=256, height=256
    )
    simple_shading_diffuse_mdl64 = BASE_CAMERA_CFG.replace(
        data_types=["simple_shading_diffuse_mdl"], width=64, height=64
    )
    simple_shading_diffuse_mdl128 = BASE_CAMERA_CFG.replace(
        data_types=["simple_shading_diffuse_mdl"], width=128, height=128
    )
    simple_shading_diffuse_mdl256 = BASE_CAMERA_CFG.replace(
        data_types=["simple_shading_diffuse_mdl"], width=256, height=256
    )
    simple_shading_full_mdl64 = BASE_CAMERA_CFG.replace(data_types=["simple_shading_full_mdl"], width=64, height=64)
    simple_shading_full_mdl128 = BASE_CAMERA_CFG.replace(data_types=["simple_shading_full_mdl"], width=128, height=128)
    simple_shading_full_mdl256 = BASE_CAMERA_CFG.replace(data_types=["simple_shading_full_mdl"], width=256, height=256)
    default = rgb64


############################


@configclass
class StateObservationCfg(dexsuite.ObservationsCfg):
    """Kuka Allegro participant scene for Dexsuite Lifting/Reorientation"""

    def __post_init__(self: dexsuite.ObservationsCfg):
        super().__post_init__()
        self.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [f"{link}_object_s" for link in FINGERTIP_LIST]},
            clip=(-20.0, 20.0),  # contact force in finger tips is under 20N normally
        )
        self.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = ["palm_link", ".*_tip"]


@configclass
class SingleCameraObservationsCfg(StateObservationCfg):
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
class DuoCameraObservationsCfg(SingleCameraObservationsCfg):
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
