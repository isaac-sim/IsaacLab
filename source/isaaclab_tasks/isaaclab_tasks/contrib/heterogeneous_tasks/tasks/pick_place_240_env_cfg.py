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
class PickPlace240EnvCfg(VLABenchmarkEnvCfg):
    """Place the spoon into the white bowl, on a table."""

    instruction = "place the spoon into the white bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.table_bamboo = library_object(
            "table_bamboo", (0.70584, 0.00721, -0.347), (0.0, 0.0, 0.707107, 0.707107), kinematic=True
        )
        self.scene.av_bowl_plate = library_object(
            "av_bowl_plate", (0.38955, -0.16128, 0.0088), (0.0, 0.0, 0.00545, 0.999985)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_03 = library_object("spatula_03", (0.3828, 0.18201, 0.01573), (0.0, 0.0, 0.365093, 0.930971))
        self.scene.empty_4 = HeterogeneousTaskManager.placeholder(4)
        self.scene.empty_5 = HeterogeneousTaskManager.placeholder(5)
        self.scene.spoon_1 = library_object("spoon_1", (0.55432, -0.04161, 0.01966), (0.0, 0.0, -0.575565, 0.817756))
        self.scene.rubiks_cube = library_object(
            "rubiks_cube", (0.50155, 0.07046, 0.04039), (0.0, 0.0, -0.660668, 0.750678)
        )
        self.scene.pitted_cherries = library_object(
            "pitted_cherries", (0.50672, 0.17329, 0.03811), (0.0, 0.0, -0.944155, 0.329503)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("spoon_1")],
                "container": SceneEntityCfg("av_bowl_plate"),
                "require_gripper_detached": True,
            },
        )
