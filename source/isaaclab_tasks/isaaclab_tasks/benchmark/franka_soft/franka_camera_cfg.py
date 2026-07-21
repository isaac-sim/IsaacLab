# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared camera and observation configs for Franka cloth/soft camera envs."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.envs import mdp as env_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg


@configclass
class FrankaTiledCameraCfg(PresetCfg):
    @configclass
    class _FrankaBaseTiledCameraCfg(CameraCfg):
        prim_path: str = "/World/envs/env_.*/Camera"
        offset: CameraCfg.OffsetCfg = CameraCfg.OffsetCfg(
            pos=(0.85, -0.55, 0.42),
            rot=(0.5080, 0.2114, 0.318, 0.7720),
            convention="opengl",
        )
        data_types: list[str] = []
        spawn: sim_utils.PinholeCameraCfg = sim_utils.PinholeCameraCfg(clipping_range=(0.01, 3.0))
        width: int = 128
        height: int = 128
        renderer_cfg: MultiBackendRendererCfg = MultiBackendRendererCfg()

    rgb = _FrankaBaseTiledCameraCfg(data_types=["rgb"])
    albedo = _FrankaBaseTiledCameraCfg(data_types=["albedo"])
    simple_shading_constant_diffuse = _FrankaBaseTiledCameraCfg(data_types=["simple_shading_constant_diffuse"])
    simple_shading_diffuse_mdl = _FrankaBaseTiledCameraCfg(data_types=["simple_shading_diffuse_mdl"])
    simple_shading_full_mdl = _FrankaBaseTiledCameraCfg(data_types=["simple_shading_full_mdl"])
    semantic_segmentation = _FrankaBaseTiledCameraCfg(data_types=["semantic_segmentation"])
    depth = _FrankaBaseTiledCameraCfg(data_types=["depth"])
    distance_to_camera = _FrankaBaseTiledCameraCfg(data_types=["distance_to_camera"])
    distance_to_image_plane = _FrankaBaseTiledCameraCfg(data_types=["distance_to_image_plane"])
    normals = _FrankaBaseTiledCameraCfg(data_types=["normals"])
    instance_segmentation_fast = _FrankaBaseTiledCameraCfg(data_types=["instance_segmentation_fast"])
    instance_id_segmentation_fast = _FrankaBaseTiledCameraCfg(data_types=["instance_id_segmentation_fast"])
    motion_vectors = _FrankaBaseTiledCameraCfg(data_types=["motion_vectors"])
    default = rgb


def camera_observations_cfg(data_type: str):
    """Build image-only observations for a camera data type.

    Args:
        data_type: Camera data type to read from the tiled camera.

    Returns:
        An observations config whose policy reads the selected camera data type.
    """

    @configclass
    class CameraObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            image = ObsTerm(
                func=env_mdp.image,
                params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": data_type, "permute": True},
            )

            def __post_init__(self) -> None:
                self.enable_corruption = False
                self.concatenate_terms = True

        policy: ObsGroup = PolicyCfg()

    return CameraObservationsCfg()
