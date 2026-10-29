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
class PickPlace063EnvCfg(VLABenchmarkEnvCfg):
    """Put the Rubik's cube on top of the can of pineapple, on a dining table."""

    instruction = "put the Rubik's cube on top of the can of pineapple"
    episode_length_s = 70.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.av_jennings_table = library_object(
            "av_jennings_table", (1.33656, 0.03581, -0.697), (0.0, 0.0, 1.0, 0.0), kinematic=True
        )
        self.scene.empty_1 = HeterogeneousTaskManager.placeholder(1)
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.spatula_05 = library_object(
            "spatula_05", (0.45306, -0.03139, 0.07471), (0.0, 0.0, 0.980569, 0.196174)
        )
        self.scene.yogurt_cup = library_object(
            "yogurt_cup", (0.43198, 0.14247, 0.09784), (0.0, 0.0, 0.989155, 0.146877)
        )
        self.scene.pineapple_slices_can = library_object(
            "pineapple_slices_can", (0.49048, 0.05821, 0.09981), (0.0, 0.0, 0.926942, 0.375205)
        )
        self.scene.av_mustard_jar = library_object(
            "av_mustard_jar", (0.38142, -0.161, 0.07108), (0.0, 0.0, 0.883734, 0.467989)
        )
        self.scene.rubiks_cube = library_object(
            "rubiks_cube", (0.51985, -0.17139, 0.10239), (0.0, 0.0, 0.689771, 0.724028)
        )
        self.scene.empty_8 = HeterogeneousTaskManager.placeholder(8)
        self.terminations.success = DoneTerm(
            func=mdp.object_on_top,
            params={
                "objects": [SceneEntityCfg("rubiks_cube")],
                "reference": SceneEntityCfg("pineapple_slices_can"),
                "require_gripper_detached": True,
            },
        )
