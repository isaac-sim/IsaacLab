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
class PickPlace050EnvCfg(VLABenchmarkEnvCfg):
    """Put the spoon and the green block in the white bowl, on a table."""

    instruction = "put the spoon and the green block in the white bowl"
    episode_length_s = 110.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_bamboo = library_object(
            "table_bamboo", (0.49651, -0.15643, -0.347), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.av_bowl_plate = library_object(
            "av_bowl_plate", (0.44455, 0.07729, 0.0088), (0.0, 0.0, -0.25256, 0.967581)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.green_block = library_object(
            "green_block", (0.3007, -0.0988, 0.0325), (0.0, 0.0, -0.120699, 0.992689)
        )
        self.scene.canned_peaches = library_object(
            "canned_peaches", (0.2933, 0.17803, 0.03789), (0.0, 0.0, 0.993811, 0.111088)
        )
        self.scene.spoon_2 = library_object("spoon_2", (0.50431, -0.15423, 0.01439), (0.0, 0.0, 0.174321, 0.984689))
        self.scene.coffee_pot = library_object(
            "coffee_pot", (0.34586, -0.22445, 0.00896), (0.0, 0.0, -0.971007, 0.239051)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("spoon_2"), SceneEntityCfg("green_block")],
                "container": SceneEntityCfg("av_bowl_plate"),
                "require_gripper_detached": True,
            },
        )
