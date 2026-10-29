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
class PickPlace223EnvCfg(VLABenchmarkEnvCfg):
    """Remove the can of mushrooms from the red bowl, on a console table."""

    instruction = "remove the can of mushrooms from the red bowl"
    episode_length_s = 50.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_eurekaconsole = library_object(
            "av_eurekaconsole", (0.41301, 0.06356, -0.697), (0.0, 0.0, -0.707107, 0.707107), kinematic=True
        )
        self.scene.bowl = library_object("bowl", (0.36179, -0.17228, 0.09648), (0.0, 0.0, 0.87799, 0.478679))
        self.scene.av_white_tray = library_object(
            "av_white_tray", (0.42322, 0.12967, 0.06906), (0.0, 0.0, -0.922862, 0.385131)
        )
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.av_chess_queen = library_object(
            "av_chess_queen", (0.53983, 0.10819, 0.069), (0.0, 0.0, 0.809415, 0.587238)
        )
        self.scene.popcorn_box = library_object(
            "popcorn_box", (0.52151, -0.12865, 0.11149), (0.0, 0.0, -0.993763, 0.111515)
        )
        self.scene.canned_mushrooms = library_object(
            "canned_mushrooms", (0.36179, -0.17228, 0.09966), (0.0, 0.0, -0.995948, 0.089933)
        )
        self.scene.av_eraser = library_object("av_eraser", (0.45311, 0.02423, 0.07333), (0.0, 0.0, -0.999416, 0.034163))
        self.scene.bbq_sauce_bottle_hot3d = library_object(
            "bbq_sauce_bottle_hot3d", (0.29973, 0.14842, 0.06906), (0.0, 0.0, -0.958274, 0.285852)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_outside_of,
            params={
                "objects": [SceneEntityCfg("canned_mushrooms")],
                "container": SceneEntityCfg("bowl"),
                "require_gripper_detached": True,
            },
        )
