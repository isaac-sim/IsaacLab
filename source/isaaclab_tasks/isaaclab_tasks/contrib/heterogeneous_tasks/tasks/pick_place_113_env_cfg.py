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
class PickPlace113EnvCfg(VLABenchmarkEnvCfg):
    """Place both the milk jug and the chess rook into the metal bin, on an end table."""

    instruction = "place both the milk jug and the chess rook into the metal bin"
    episode_length_s = 110.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_ellendale = library_object(
            "av_ellendale", (0.39882, 0.03972, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.bin_a05 = library_object("bin_a05", (0.40658, 0.04101, -0.001), (0.0, 0.0, -0.999895, 0.014505))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.oatmeal_raisin_cookies = library_object(
            "oatmeal_raisin_cookies", (0.50615, -0.14806, 0.08262), (0.0, 0.0, -0.504243, 0.863562)
        )
        self.scene.milkjug_a01 = library_object(
            "milkjug_a01", (0.45731, 0.23451, -0.001), (0.0, 0.0, -0.175617, 0.984459)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.av_chess_rook = library_object(
            "av_chess_rook", (0.30069, -0.16045, -0.001), (0.0, 0.0, 0.271419, 0.962461)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("milkjug_a01"), SceneEntityCfg("av_chess_rook")],
                "container": SceneEntityCfg("bin_a05"),
                "require_gripper_detached": True,
            },
        )
