# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The G1 apple pick-and-place task with a head camera image that Cosmos generates from depth."""

import os

from isaaclab.utils import configclass

from isaaclab_contrib.sensors.cosmos_transfer import (
    CosmosTransferClientCfg,
    CosmosTransferProcessorCfg,
    cosmos_camera,
    episode_frame_budget,
)

from .apple_pick_place_g1_dex3_env_cfg import HEAD_CAMERA_CFG, ApplePickPlaceSceneCfg, G1ApplePickPlaceEnvCfg

# Appearance of the real demonstrations, so the generated images look like the images the policy was trained on.
APPLE_COSMOS_PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompts", "apple_pick_place.json")

# The hands, apple and plate lie between 0.15 m and 0.9 m from the head camera.
APPLE_COSMOS_HEAD_CAMERA_CFG = cosmos_camera(
    HEAD_CAMERA_CFG,
    CosmosTransferClientCfg(prompt_files=APPLE_COSMOS_PROMPT_FILE),
    near=0.15,
    far=0.9,
    name="head_camera",
)


@configclass
class G1ApplePickPlaceCosmosEnvCfg(G1ApplePickPlaceEnvCfg):
    """The apple task with Cosmos generating the head camera's image.

    Registered as ``IsaacContrib-PickPlace-Apple-G1-Dex3-Cosmos``. Start the Cosmos service first.
    """

    scene: ApplePickPlaceSceneCfg = ApplePickPlaceSceneCfg(
        num_envs=1, env_spacing=3.0, head_camera=APPLE_COSMOS_HEAD_CAMERA_CFG
    )

    @property
    def cosmos(self) -> CosmosTransferClientCfg:
        """The head camera's Cosmos settings; overrides such as ``env.cosmos.prompt_files=...`` set them."""
        return self._cosmos_transfer().backend

    @cosmos.setter
    def cosmos(self, settings: CosmosTransferClientCfg):
        # configclass copies properties like fields when it copies a configuration, which needs a setter.
        self._cosmos_transfer().backend = settings

    @property
    def cosmos_seed(self) -> int | list[int]:
        """Cosmos seed of environment 0, or a list with one seed per environment; overrides such as
        ``env.cosmos_seed=[101,202]`` set it on the head camera's Cosmos step."""
        return self._cosmos_transfer().seed

    @cosmos_seed.setter
    def cosmos_seed(self, seed: int | list[int]):
        self._cosmos_transfer().seed = seed

    def validate_config(self):
        """Size the Cosmos episode to the task's final duration and capture rate."""
        if self.cosmos.max_episode_frames is None:
            self.cosmos.max_episode_frames = episode_frame_budget(self, self.scene.head_camera.update_period)[2]

    def _cosmos_transfer(self) -> CosmosTransferProcessorCfg:
        """Return the head camera's Cosmos step."""
        processors = self.scene.head_camera.tensor_processors
        return next(cfg for cfg in processors if isinstance(cfg, CosmosTransferProcessorCfg))
