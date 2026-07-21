# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark camera variant of the Franka volume deformable lifting environment."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import (
    FrankaSoftEnvCfg,
    _FrankaSoftSceneCfg,
)
from isaaclab_tasks.utils import PresetCfg

from .franka_camera_cfg import FrankaTiledCameraCfg, camera_observations_cfg


@configclass
class _FrankaSoftCameraSceneCfg(_FrankaSoftSceneCfg):
    """Franka soft scene with a tiled camera sensor."""

    tiled_camera: FrankaTiledCameraCfg = FrankaTiledCameraCfg()


@configclass
class FrankaSoftCameraSceneCfg(PresetCfg):
    """Scene presets for the Franka soft camera benchmark task."""

    newton_mjwarp_vbd: _FrankaSoftCameraSceneCfg = _FrankaSoftCameraSceneCfg(
        num_envs=4, env_spacing=3.0, replicate_physics=True
    )
    newton_mjwarp_vbd_proxy = newton_mjwarp_vbd
    # PhysX does not support replicating physics for deformable objects.
    physx: _FrankaSoftCameraSceneCfg = _FrankaSoftCameraSceneCfg(num_envs=4, env_spacing=3.0, replicate_physics=False)
    default = newton_mjwarp_vbd_proxy


@configclass
class FrankaSoftCameraEnvCfg(PresetCfg):
    """Benchmark-only Franka soft camera environments, one per rendered data type."""

    @configclass
    class BaseFrankaSoftCameraEnvCfg(FrankaSoftEnvCfg):
        """Camera variant of the Franka soft lift environment."""

        scene: FrankaSoftCameraSceneCfg = FrankaSoftCameraSceneCfg()

        def __post_init__(self) -> None:
            super().__post_init__()
            self.commands.deformable_pose.debug_vis = False
            self.events.reset_deformable.params["position_range"] = {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
            }

    rgb = BaseFrankaSoftCameraEnvCfg(observations=camera_observations_cfg("rgb"))
    albedo = BaseFrankaSoftCameraEnvCfg(observations=camera_observations_cfg("albedo"))
    simple_shading_constant_diffuse = BaseFrankaSoftCameraEnvCfg(
        observations=camera_observations_cfg("simple_shading_constant_diffuse")
    )
    simple_shading_diffuse_mdl = BaseFrankaSoftCameraEnvCfg(
        observations=camera_observations_cfg("simple_shading_diffuse_mdl")
    )
    simple_shading_full_mdl = BaseFrankaSoftCameraEnvCfg(
        observations=camera_observations_cfg("simple_shading_full_mdl")
    )
    semantic_segmentation = BaseFrankaSoftCameraEnvCfg(observations=camera_observations_cfg("semantic_segmentation"))
    depth = BaseFrankaSoftCameraEnvCfg(observations=camera_observations_cfg("depth"))
    distance_to_camera = BaseFrankaSoftCameraEnvCfg(observations=camera_observations_cfg("distance_to_camera"))
    distance_to_image_plane = BaseFrankaSoftCameraEnvCfg(
        observations=camera_observations_cfg("distance_to_image_plane")
    )
    normals = BaseFrankaSoftCameraEnvCfg(observations=camera_observations_cfg("normals"))
    instance_segmentation_fast = BaseFrankaSoftCameraEnvCfg(
        observations=camera_observations_cfg("instance_segmentation_fast")
    )
    instance_id_segmentation_fast = BaseFrankaSoftCameraEnvCfg(
        observations=camera_observations_cfg("instance_id_segmentation_fast")
    )
    motion_vectors = BaseFrankaSoftCameraEnvCfg(observations=camera_observations_cfg("motion_vectors"))
    default = rgb
