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
class PickPlace237EnvCfg(VLABenchmarkEnvCfg):
    """Put the barbecue sauce bottle in the metal bowl, on a storage unit."""

    instruction = "put the barbecue sauce bottle in the metal bowl"
    episode_length_s = 60.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.standard_smallunit = library_object(
            "standard_smallunit", (0.61388, 0.15443, -0.6376), (0.0, 0.0, 0.0, 1.0), kinematic=True
        )
        self.scene.serving_bowl = library_object(
            "serving_bowl", (0.43431, 0.04008, 0.15625), (0.0, 0.0, -0.382123, 0.924111)
        )
        self.scene.empty_2 = HeterogeneousTaskManager.placeholder(2)
        self.scene.empty_3 = HeterogeneousTaskManager.placeholder(3)
        self.scene.butter = library_object("butter", (0.31292, 0.16213, 0.18241), (0.0, 0.0, 0.666006, 0.745946))
        self.scene.soup_can = library_object("soup_can", (0.31203, -0.07995, 0.15606), (0.0, 0.0, 0.846199, 0.532866))
        self.scene.spoon_big = library_object(
            "spoon_big", (0.38302, -0.26315, 0.15624), (0.0, 0.0, -0.997833, 0.065802)
        )
        self.scene.tomato_sauce_can = library_object(
            "tomato_sauce_can", (0.42259, 0.24656, 0.19742), (0.0, 0.0, -0.102168, 0.994767)
        )
        self.scene.bbq_sauce_bottle = library_object(
            "bbq_sauce_bottle", (0.46072, -0.15377, 0.23016), (0.0, 0.0, 0.810116, 0.58627)
        )
        self.terminations.success = DoneTerm(
            func=mdp.object_in_container,
            params={
                "objects": [SceneEntityCfg("bbq_sauce_bottle")],
                "container": SceneEntityCfg("serving_bowl"),
                "require_gripper_detached": True,
            },
        )
