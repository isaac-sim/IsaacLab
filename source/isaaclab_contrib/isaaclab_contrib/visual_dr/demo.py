# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in config clone for inspecting the observation boundary; not an RL benchmark."""

from isaaclab.managers import ObservationTermCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.stack.config.franka.stack_ik_rel_visuomotor_cosmos_env_cfg import (
    FrankaCubeStackVisuomotorCosmosEnvCfg,
)

from .observations import image_runtime_dr


@configclass
class FrankaStackRuntimeDRCfg(FrankaCubeStackVisuomotorCosmosEnvCfg):
    """One environment, two cameras, raw semantic IDs; existing task is untouched."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        for name in ("table_cam", "wrist_cam"):
            camera = getattr(self.scene, name)
            camera.data_types = ["rgb", "distance_to_image_plane", "semantic_segmentation"]
            camera.renderer_cfg.colorize_semantic_segmentation = False
            camera.renderer_cfg.semantic_segmentation_mapping = {}
            setattr(self.observations.policy, name, ObservationTermCfg(func=image_runtime_dr, params={"camera": name}))
        for name in ("table_cam_segmentation", "table_cam_normals", "table_cam_depth"):
            setattr(self.observations.policy, name, None)
