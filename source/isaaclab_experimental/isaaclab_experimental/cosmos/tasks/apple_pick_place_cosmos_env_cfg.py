# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The G1 apple pick-and-place task with a head camera image that Cosmos generates from depth."""

import json
import os

from isaaclab.utils import configclass

from isaaclab_tasks.contrib.apple_pick_place.apple_pick_place_g1_dex3_env_cfg import (
    HEAD_CAMERA_CFG,
    ApplePickPlaceSceneCfg,
    G1ApplePickPlaceEnvCfg,
)

from .. import CosmosModelCfg, CosmosTransferProcessorCfg, cosmos_camera, episode_frame_budget

# Appearance of the real demonstrations, so the generated images look like the images the policy was trained on.
APPLE_COSMOS_PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompts", "apple_pick_place.json")

APPLE_COSMOS_DEPTH_DIRECTIVE = (
    " Follow the depth control video precisely: shape, contour, silhouette, position, and motion of every visible"
    " structure must align with the depth signal at every frame."
)
"""Sentence after the caption asking Cosmos to keep the depth's geometry and motion."""

APPLE_COSMOS_CAPTURE_STEPS = 4
"""Policy steps per Cosmos depth image: 7.5 Hz at the task's 30 Hz. One 4-frame Cosmos update then spans one 16-step
action chunk of the policy."""


def apple_cosmos_prompt() -> str:
    """The caption as compact JSON followed by the depth directive, the prompt of the earlier evaluation."""
    with open(APPLE_COSMOS_PROMPT_FILE, encoding="utf-8") as file:
        caption = json.load(file)
    return json.dumps(caption, ensure_ascii=False, separators=(",", ":")) + APPLE_COSMOS_DEPTH_DIRECTIVE


# The hands, apple and plate lie between 0.15 m and 0.9 m from the head camera. The prompt, the 12.5 fps the model is
# told, and the capture rate below follow the earlier Isaac Sim evaluation of this policy.
APPLE_COSMOS_HEAD_CAMERA_CFG = cosmos_camera(
    HEAD_CAMERA_CFG,
    CosmosModelCfg(prompt=apple_cosmos_prompt(), fps=12.5),
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

    def __post_init__(self):
        super().__post_init__()
        self.scene.head_camera.update_period = APPLE_COSMOS_CAPTURE_STEPS * self.sim.dt * self.decimation

    def validate_config(self):
        """Size the Cosmos episode to the task's final duration and capture rate."""
        processors = self.scene.head_camera.tensor_processors
        model = next(cfg.backend for cfg in processors if isinstance(cfg, CosmosTransferProcessorCfg))
        if model.max_episode_frames is None:
            model.max_episode_frames = episode_frame_budget(self, self.scene.head_camera.update_period)[2]
