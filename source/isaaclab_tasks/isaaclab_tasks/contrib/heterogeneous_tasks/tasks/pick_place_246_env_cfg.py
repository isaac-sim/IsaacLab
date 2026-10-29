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
class PickPlace246EnvCfg(VLABenchmarkEnvCfg):
    """Pick up the chess pawn and put it in the black bowl, on a dining table."""

    instruction = "pick up the chess pawn and put it in the black bowl"
    episode_length_s = 55.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_desperes_table = library_object(
            "av_desperes_table", (0.57621, 0.43, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.blackandbrassbowl_large = library_object(
            "blackandbrassbowl_large", (0.43497, -0.00449, 0.07694), (0.0, 0.0, -0.682002, 0.73135)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.av_chess_pawn = library_object(
            "av_chess_pawn", (0.35685, -0.2027, 0.077), (0.0, 0.0, 0.397104, 0.917774)
        )
        self.scene.orange_juice_carton = library_object(
            "orange_juice_carton", (0.52407, -0.21711, 0.17324), (0.0, 0.0, 0.799739, 0.600348)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.green_beans_can = library_object(
            "green_beans_can", (0.33247, 0.26328, 0.10579), (0.0, 0.0, 0.70727, 0.706944)
        )
        self.scene.brick = library_object("brick", (0.44713, 0.19841, 0.10257), (0.0, 0.0, 0.775968, 0.630772))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("av_chess_pawn")],
                "container": SceneEntityCfg("blackandbrassbowl_large"),
                "require_gripper_detached": True,
            },
        )
