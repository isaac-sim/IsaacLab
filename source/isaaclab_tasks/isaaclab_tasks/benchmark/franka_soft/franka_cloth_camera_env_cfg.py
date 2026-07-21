# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark camera variant of the Franka surface deformable lifting environment."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.lift.config.franka_soft.franka_cloth_env_cfg import (
    FrankaClothEnvCfg,
    FrankaClothSceneCfg,
)
from isaaclab_tasks.utils import PresetCfg

from .franka_camera_cfg import FrankaTiledCameraCfg, camera_observations_cfg


@configclass
class FrankaClothCameraSceneCfg(FrankaClothSceneCfg):
    """Franka cloth scene with a tiled camera sensor."""

    tiled_camera: FrankaTiledCameraCfg = FrankaTiledCameraCfg()


@configclass
class FrankaClothCameraEnvCfg(PresetCfg):
    """Benchmark-only Franka cloth camera environments, one per rendered data type."""

    @configclass
    class BaseFrankaClothCameraEnvCfg(FrankaClothEnvCfg):
        """Camera variant of the Franka cloth lift environment."""

        scene = FrankaClothCameraSceneCfg(num_envs=4, env_spacing=3.0, replicate_physics=True)

        def __post_init__(self) -> None:
            super().__post_init__()
            self.commands.deformable_pose.debug_vis = False
            self.events.reset_deformable.params["position_range"] = {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
            }

    rgb = BaseFrankaClothCameraEnvCfg(observations=camera_observations_cfg("rgb"))
    albedo = BaseFrankaClothCameraEnvCfg(observations=camera_observations_cfg("albedo"))
    simple_shading_constant_diffuse = BaseFrankaClothCameraEnvCfg(
        observations=camera_observations_cfg("simple_shading_constant_diffuse")
    )
    simple_shading_diffuse_mdl = BaseFrankaClothCameraEnvCfg(
        observations=camera_observations_cfg("simple_shading_diffuse_mdl")
    )
    simple_shading_full_mdl = BaseFrankaClothCameraEnvCfg(
        observations=camera_observations_cfg("simple_shading_full_mdl")
    )
    semantic_segmentation = BaseFrankaClothCameraEnvCfg(observations=camera_observations_cfg("semantic_segmentation"))
    depth = BaseFrankaClothCameraEnvCfg(observations=camera_observations_cfg("depth"))
    distance_to_camera = BaseFrankaClothCameraEnvCfg(observations=camera_observations_cfg("distance_to_camera"))
    distance_to_image_plane = BaseFrankaClothCameraEnvCfg(
        observations=camera_observations_cfg("distance_to_image_plane")
    )
    normals = BaseFrankaClothCameraEnvCfg(observations=camera_observations_cfg("normals"))
    instance_segmentation_fast = BaseFrankaClothCameraEnvCfg(
        observations=camera_observations_cfg("instance_segmentation_fast")
    )
    instance_id_segmentation_fast = BaseFrankaClothCameraEnvCfg(
        observations=camera_observations_cfg("instance_id_segmentation_fast")
    )
    motion_vectors = BaseFrankaClothCameraEnvCfg(observations=camera_observations_cfg("motion_vectors"))
    default = rgb
