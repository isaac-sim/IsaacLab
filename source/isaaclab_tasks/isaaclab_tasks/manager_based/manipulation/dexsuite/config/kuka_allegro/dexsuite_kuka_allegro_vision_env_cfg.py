# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns, MultiMeshRayCasterCameraCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ... import dexsuite_env_cfg as dexsuite_state_impl
from ... import mdp
from .dexsuite_kuka_allegro_env_cfg import KukaAllegroMixinCfg


@configclass
class KukaAllegroRayCasterSceneCfg(dexsuite_state_impl.SceneCfg):
    """Dexsuite scene for multi-objects Lifting/Reorientation"""

    base_camera = MultiMeshRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/root_joint",
        mesh_prim_paths=[
            "/World/GroundPlane",
            MultiMeshRayCasterCfg.RaycastTargetCfg(is_shared=False, target_prim_expr="{ENV_REGEX_NS}/Object"),
            MultiMeshRayCasterCfg.RaycastTargetCfg(is_shared=True, target_prim_expr="{ENV_REGEX_NS}/Robot/ee_link/.*_link.*"),
            MultiMeshRayCasterCfg.RaycastTargetCfg(is_shared=True, target_prim_expr="{ENV_REGEX_NS}/Robot/.*_link_.*"),
            MultiMeshRayCasterCfg.RaycastTargetCfg(is_shared=True, target_prim_expr="{ENV_REGEX_NS}/table", track_mesh_transforms=False),
        ],
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
            pos=(0.57, -0.8, 0.5), rot=(0.6124, 0.6124, 0.3536, 0.3536), convention="opengl",
        ),
        pattern_cfg=patterns.PinholeCameraPatternCfg(height=64, width=64),
        max_distance=10,
        depth_clipping_behavior="max"
    )

    wrist_camera = MultiMeshRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ee_link/palm_link",
        mesh_prim_paths=[
            "/World/GroundPlane",
            MultiMeshRayCasterCfg.RaycastTargetCfg(is_shared=False, target_prim_expr="{ENV_REGEX_NS}/Object"),
            MultiMeshRayCasterCfg.RaycastTargetCfg(is_shared=True, target_prim_expr="{ENV_REGEX_NS}/Robot/ee_link/.*_link.*"),
            MultiMeshRayCasterCfg.RaycastTargetCfg(is_shared=True, target_prim_expr="{ENV_REGEX_NS}/table"),
        ],
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
            pos=(0.038, -0.38, -0.18), rot=(0.299, 0.641, 0.641, -0.299), convention="opengl",
        ),
        pattern_cfg=patterns.PinholeCameraPatternCfg(height=64, width=64),
        max_distance=10,
        depth_clipping_behavior="max"
    )


@configclass
class KukaAllegroTiledCameraSceneCfg(dexsuite_state_impl.SceneCfg):
    """Dexsuite scene for multi-objects Lifting/Reorientation"""

    base_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.57, -0.8, 0.5), rot=(0.6124, 0.6124, 0.3536, 0.3536), convention="opengl",
        ),
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(),
        width=64,
        height=64,
    )

    wrist_camera = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/ee_link/palm_link/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.038, -0.38, -0.18), rot=(0.299, 0.641, 0.641, -0.299), convention="opengl",
        ),
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(),
        width=64,
        height=64,
    )


@configclass
class KukaAllegroCameraObservationsCfg(dexsuite_state_impl.ObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class BaseImageObsCfg(ObsGroup):
        """Camera observations for policy group."""

        object_observation_b = ObsTerm(
            func=mdp.depth_image,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-1.0, 1.0),
            params={"sensor_cfg": SceneEntityCfg("base_camera")},
        )

    @configclass
    class WristImageObsCfg(ObsGroup):
        wrist_observation = ObsTerm(
            func=mdp.depth_image,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-1.0, 1.0),
            params={"sensor_cfg": SceneEntityCfg("wrist_camera")},
        )

    base_image: BaseImageObsCfg = BaseImageObsCfg()
    wrist_image: WristImageObsCfg = WristImageObsCfg()


@configclass
class KukaAllegroDepthTiledCameraMixinCfg(KukaAllegroMixinCfg):
    scene: KukaAllegroTiledCameraSceneCfg = KukaAllegroTiledCameraSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)
    observations: KukaAllegroCameraObservationsCfg = KukaAllegroCameraObservationsCfg()


@configclass
class KukaAllegroDepthRayCasterCameraMixinCfg(KukaAllegroMixinCfg):
    scene: KukaAllegroRayCasterSceneCfg = KukaAllegroRayCasterSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)
    observations: KukaAllegroCameraObservationsCfg = KukaAllegroCameraObservationsCfg()


@configclass
class DexsuiteKukaAllegroLiftDepthTiledCameraEnvCfg(KukaAllegroDepthTiledCameraMixinCfg, dexsuite_state_impl.DexsuiteLiftEnvCfg):
    pass


@configclass
class DexsuiteKukaAllegroLiftDepthTiledCameraEnvCfg_PLAY(KukaAllegroDepthTiledCameraMixinCfg, dexsuite_state_impl.DexsuiteLiftEnvCfg_PLAY):
    pass


@configclass
class DexsuiteKukaAllegroLiftDepthRayCasterCameraEnvCfg(KukaAllegroDepthRayCasterCameraMixinCfg, dexsuite_state_impl.DexsuiteLiftEnvCfg):
    pass


@configclass
class DexsuiteKukaAllegroLiftDepthRayCasterCameraEnvCfg_PLAY(KukaAllegroDepthRayCasterCameraMixinCfg, dexsuite_state_impl.DexsuiteLiftEnvCfg_PLAY):
    pass
