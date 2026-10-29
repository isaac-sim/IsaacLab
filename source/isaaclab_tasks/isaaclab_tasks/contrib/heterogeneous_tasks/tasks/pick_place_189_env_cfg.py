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
class PickPlace189EnvCfg(VLABenchmarkEnvCfg):
    """Pick up the tomato soup can and put it in the red bowl, on an end table."""

    instruction = "pick up the tomato soup can and put it in the red bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.appleseed_endtable = library_object(
            "appleseed_endtable", (0.42556, -0.00621, -0.697), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.bowl = library_object("bowl", (0.43385, -0.1056, -0.02852), (0.0, 0.0, 0.95839, 0.285462))
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.av_chess_queen = library_object(
            "av_chess_queen", (0.38115, 0.11379, -0.056), (0.0, 0.0, 0.392464, 0.919767)
        )
        self.scene.tomato_soup_can = library_object(
            "tomato_soup_can", (0.53042, 0.1707, -0.00498), (0.0, 0.0, -0.112382, 0.993665)
        )
        self.scene.empty_6 = HeterogeneousTaskManager.placeholder(6)
        self.scene.av_woodenchsmall = library_object(
            "av_woodenchsmall", (0.29521, 0.06338, -0.056), (0.0, 0.0, -0.999988, 0.004988)
        )
        self.scene.red_block = library_object("red_block", (0.5743, -0.03415, -0.0325), (0.0, 0.0, 0.930398, 0.36655))
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("tomato_soup_can")],
                "container": SceneEntityCfg("bowl"),
                "require_gripper_detached": True,
            },
        )
