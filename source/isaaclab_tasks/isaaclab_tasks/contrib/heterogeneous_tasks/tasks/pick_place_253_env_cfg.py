# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import HeterogeneousTaskManager, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .. import mdp
from ..heterogeneous_tasks_env_cfg import VLABenchmarkEnvCfg
from ..objects import library_object


@configclass
class PickPlace253EnvCfg(VLABenchmarkEnvCfg):
    """Put a chess piece in the metal bin, on a table."""

    instruction = "put a chess piece in the metal bin"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_whittershins = library_object(
            "av_whittershins", (0.92018, 0.01698, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_a03 = library_object("bin_a03", (0.36708, -0.02931, -0.058), (0.0, 0.0, 0.769312, 0.638873))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_09 = library_object(
            "spatula_09", (0.52894, 0.01911, -0.05127), (0.0, 0.0, 0.756174, 0.654371)
        )
        self.scene.av_chess_queen = library_object(
            "av_chess_queen", (0.48813, 0.27045, -0.058), (0.0, 0.0, 0.978983, 0.203942)
        )
        self.scene.granola_bars = library_object(
            "granola_bars", (0.35543, 0.25818, 0.004), (0.0, 0.0, 0.790531, 0.612421)
        )
        self.scene.av_chess_bishop = library_object(
            "av_chess_bishop", (0.30548, -0.15134, -0.058), (0.0, 0.0, -0.264638, 0.964348)
        )
        self.scene.frozen_vegetable_block = library_object(
            "frozen_vegetable_block", (0.48385, -0.17023, -0.058), (0.0, 0.0, 0.37929, 0.925278)
        )
        self.scene.brick = library_object("brick", (0.41087, 0.13708, -0.03243), (0.0, 0.0, 0.937211, 0.348762))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("av_chess_bishop"), SceneEntityCfg("av_chess_queen")],
                "container": SceneEntityCfg("bin_a03"),
                "logical": "any",
                "require_gripper_detached": True,
            },
        )
