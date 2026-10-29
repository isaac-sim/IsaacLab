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
class PickPlace169EnvCfg(VLABenchmarkEnvCfg):
    """Put a chess piece in the metal bin, on a console table."""

    instruction = "put a chess piece in the metal bin"
    episode_length_s = 65.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_eurekaconsole = library_object(
            "av_eurekaconsole", (0.41264, -0.19095, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_a05 = library_object("bin_a05", (0.42961, 0.08486, 0.069), (0.0, 0.0, -0.562536, 0.826773))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.av_kiwi01 = library_object("av_kiwi01", (0.39355, -0.26022, 0.06899), (0.0, 0.0, 0.998405, 0.05646))
        self.scene.av_chess_pawn = library_object(
            "av_chess_pawn", (0.44288, -0.14442, 0.069), (0.0, 0.0, -0.984048, 0.177902)
        )
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.av_chess_bishop = library_object(
            "av_chess_bishop", (0.47694, -0.21928, 0.069), (0.0, 0.0, 0.955219, 0.295899)
        )
        self.scene.tomato_sauce_can = library_object(
            "tomato_sauce_can", (0.30149, -0.12374, 0.11042), (0.0, 0.0, 0.99937, 0.035496)
        )
        self.scene.macaroni_and_cheese = library_object(
            "macaroni_and_cheese", (0.53821, -0.14886, 0.15213), (0.0, 0.0, 0.077179, 0.997017)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("av_chess_bishop"), SceneEntityCfg("av_chess_pawn")],
                "container": SceneEntityCfg("bin_a05"),
                "logical": "any",
                "require_gripper_detached": True,
            },
        )
