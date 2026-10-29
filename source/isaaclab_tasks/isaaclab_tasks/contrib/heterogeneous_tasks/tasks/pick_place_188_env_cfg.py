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
class PickPlace188EnvCfg(VLABenchmarkEnvCfg):
    """Pick up the popcorn box and put it in the black bowl, on a console table."""

    instruction = "pick up the popcorn box and put it in the black bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_ellisville = library_object(
            "av_ellisville", (0.41725, -0.3236, -0.69599), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.blackandbrassbowl_small = library_object(
            "blackandbrassbowl_small", (0.42062, 0.03071, 0.19796), (0.0, 0.0, -0.554522, 0.832169)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.av_chess_queen = library_object(
            "av_chess_queen", (0.52846, -0.0701, 0.198), (0.0, 0.0, -0.998332, 0.057727)
        )
        self.scene.popcorn_box = library_object(
            "popcorn_box", (0.47733, -0.18862, 0.24049), (0.0, 0.0, 0.406695, 0.913564)
        )
        self.scene.peas_and_carrots = library_object(
            "peas_and_carrots", (0.49413, 0.23196, 0.22726), (0.0, 0.0, -0.989803, 0.142442)
        )
        self.scene.coffee_pot = library_object(
            "coffee_pot", (0.33965, 0.17579, 0.19796), (0.0, 0.0, 0.265764, 0.964038)
        )
        self.scene.coffee_can = library_object(
            "coffee_can", (0.33054, -0.2261, 0.26807), (0.0, 0.0, 0.870916, 0.491432)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("popcorn_box")],
                "container": SceneEntityCfg("blackandbrassbowl_small"),
                "require_gripper_detached": True,
            },
        )
